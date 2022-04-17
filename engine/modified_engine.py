import os
import sys
import cv2
import time
import torch
import logging
import contextlib
import numpy as np
import torch.nn as nn
import detectron2.utils.comm as comm
import detectron2.data.transforms as T

from typing import Union, List
from collections import OrderedDict
from detectron2.engine.train_loop import TrainerBase
from detectron2.config import CfgNode
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    MetadataCatalog
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    COCOEvaluator
)
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, hooks
from detectron2.modeling import build_model
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.solver import build_optimizer, build_lr_scheduler
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from engine import BasePredictor
from utils.hook import LossEvalHook
from utils.default_config import _C
from utils.model_utils import transfer_weight
from utils.data_utils import build_mapper, get_noise_adder, load_image, split_image_into_slices, merge_slices
from model import Noise2Void, ModifiedRCNN


try:
    _nullcontext = contextlib.nullcontext
except AttributeError:

    @contextlib.contextmanager
    def _nullcontext(enter_result=None):
        yield enter_result


class SimpleEngine(TrainerBase):
    def __init__(self,
                 main_model: torch.nn.Module,
                 n2v: torch.nn.Module,
                 main_optimizer,
                 n2v_optimizer,
                 data_loader):

        super().__init__()

        main_model.train()
        n2v.train()

        # Model
        self.main_model = main_model        # Noise2Void + Mask R-CNN
        self.n2v = n2v                      # Noise2Void

        # Optimizer
        self.main_optimizer = main_optimizer
        self.n2v_optimizer = n2v_optimizer

        # Data Loader
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)


    def run_step(self):
        assert self.main_model.training, "Model 'main_model' is not in training mode."
        assert self.n2v.training, "Model 'n2v' is not in training mode."

        # Measure Data Time
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        # Update Noise2Void
        self.n2v_optimizer.zero_grad()
        n2v_loss = self.n2v(data)
        n2v_loss.backward()
        self.n2v_optimizer.step()

        # Transfer Weight and Update Main Model
        transfer_weight(src_model=self.n2v.model, dst_model=self.main_model.N2VNet)
        self.freeze_model(model=self.main_model.N2VNet)
        self.main_optimizer.zero_grad()

        # Calculate Total Losses
        loss_dict = self.main_model(data)
        losses = sum(loss_dict.values())
        losses.backward()

        with torch.cuda.stream(
            torch.cuda.Stream()
        ) if losses.device.type == "cuda" else _nullcontext():

            metrics_dict = {}
            metrics_dict.update({'n2v_loss': n2v_loss})
            metrics_dict.update(loss_dict)
            metrics_dict.update({'data_time': data_time})

            self._write_metrics(metrics_dict)
            self._detect_anomaly(n2v_loss)
            self._detect_anomaly(losses)

        self.main_optimizer.step()

    def freeze_model(self, model: nn.Module, inverse: bool = False):
        for param in model.parameters():
            if not inverse: param.requires_grad = False
            else: param.requires_grad = True

    def _detect_anomaly(self, losses):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\n".format(self.iter))

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(loss for loss in metrics_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)


class DefaultEngine(SimpleEngine):
    def __init__(self, cfg: CfgNode):
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        model = self.build_model(cfg)
        n2v = Noise2Void(cfg).to(cfg.MODEL.DEVICE)

        main_optimizer = self.build_main_optimizer(cfg, model)
        n2v_optimizer = self.build_n2v_optimizer(cfg, n2v)
        data_loader = self.build_train_loader(cfg)

        super().__init__(main_model=model,
                         n2v=n2v,
                         main_optimizer=main_optimizer,
                         n2v_optimizer=n2v_optimizer,
                         data_loader=data_loader)

        self.scheduler = self.build_lr_scheduler(cfg, main_optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            main_optimizer=main_optimizer,
            n2v_optimizer=n2v_optimizer,
            scheduler=self.scheduler)
        
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume: bool = True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [hooks.IterationTimer(), hooks.LRScheduler(self.main_optimizer, self.scheduler)]
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.main_model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def build_writers(self):
        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    def train(self):
        super().train(self.start_iter, self.max_iter)

    @classmethod
    def build_model(cls, cfg: CfgNode):
        model = build_model(cfg)
        return model

    @classmethod
    def build_main_optimizer(cls, cfg: CfgNode, main_model: nn.Module):
        return build_optimizer(cfg, main_model)

    @classmethod
    def build_n2v_optimizer(cls, cfg: CfgNode, n2v_model: nn.Module):
        return torch.optim.Adam(params=n2v_model.parameters(), lr=cfg.MODEL.N2V.LR, betas=(0.5, 0.999))

    @classmethod
    def build_lr_scheduler(cls, cfg: CfgNode, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        train_mapper = build_mapper(resize=cfg.INPUT.RESIZE,
                                    noise_type=cfg.INPUT.NOISE_TYPE,
                                    noise_param=cfg.INPUT.NOISE_PARAM,
                                    use_n2v=True)
        return build_detection_train_loader(cfg, mapper=train_mapper)

    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name: str):
        val_mapper = build_mapper(resize=cfg.INPUT.RESIZE, noise_type='none',  use_n2v=True)
        return build_detection_test_loader(cfg, dataset_name, mapper=val_mapper)

    @classmethod
    def build_evaluator(cls, cfg: CfgNode, dataset_name: str):
        raise NotImplementedError()

    @classmethod
    def test(cls, cfg: CfgNode, model: nn.Module, evaluators=None):
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warning(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @staticmethod
    def auto_scale_workers(cfg, num_workers: int):
        old_world_size = cfg.SOLVER.REFERENCE_WORLD_SIZE
        if old_world_size == 0 or old_world_size == num_workers:
            return cfg
        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        assert (
            cfg.SOLVER.IMS_PER_BATCH % old_world_size == 0
        ), "Invalid REFERENCE_WORLD_SIZE in config!"
        scale = num_workers / old_world_size
        bs = cfg.SOLVER.IMS_PER_BATCH = int(round(cfg.SOLVER.IMS_PER_BATCH * scale))
        lr = cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * scale
        max_iter = cfg.SOLVER.MAX_ITER = int(round(cfg.SOLVER.MAX_ITER / scale))
        warmup_iter = cfg.SOLVER.WARMUP_ITERS = int(round(cfg.SOLVER.WARMUP_ITERS / scale))
        cfg.SOLVER.STEPS = tuple(int(round(s / scale)) for s in cfg.SOLVER.STEPS)
        cfg.TEST.EVAL_PERIOD = int(round(cfg.TEST.EVAL_PERIOD / scale))
        cfg.SOLVER.REFERENCE_WORLD_SIZE = num_workers  # maintain invariant
        logger = logging.getLogger(__name__)
        logger.info(
            f"Auto-scaling the config to batch_size={bs}, learning_rate={lr}, "
            f"max_iter={max_iter}, warmup={warmup_iter}."
        )

        if frozen:
            cfg.freeze()
        return cfg


class DMRCNNTrainer(DefaultEngine):
    @classmethod
    def build_evaluator(cls, cfg: CfgNode, dataset_name: str, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        val_mapper = build_mapper(resize=self.cfg.INPUT.RESIZE, noise_type='none', use_n2v=True)
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.main_model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                mapper=val_mapper)))
        return hooks


class DMRCNNPredictor(BasePredictor):
    def __init__(self, cfg_file: str, weight_file: str, score_thres: float = 0.60):
        super().__init__(cfg_file, weight_file, score_thres)
        self.denoiser = self.model.N2VNet
        
        print(f"* Predictor '{self.__class__.__name__}' is initialized.")
        print(f"    - Configuration: '{cfg_file}'")
        print(f"    - Weight: '{weight_file}'")
        print(f"    - Confidence Threshold: {score_thres}")

    def __call__(self, origin_img: np.ndarray):
        predictions, inputs = self._base_call(origin_img)
        
        # Get Denoised Image
        image, height, width = inputs['image'], inputs['height'], inputs['width']
        
        denoised_img = ((image - 0.5) / 0.5).unsqueeze(0)
        denoised_img = self.model.N2VNet(denoised_img.to(self.device))
        denoised_img = (((denoised_img * 0.5) + 0.5).clamp(0.0, 1.0) * 255)[0].permute(1, 2, 0).detach().cpu().numpy()
        denoised_img = cv2.resize(denoised_img, dsize=(height, width))
        return predictions, origin_img, denoised_img
    
    def inference_denoiser(self, image_file: str, noise_type: str, noise_params: List[Union[int, float]], save_path: str) -> np.ndarray:
        image_arr = load_image(image_file)
        noise_maker = get_noise_adder(noise_type, noise_params)
            
        d_h, d_w = image_arr.shape[:2]
        image_arr = self.aug.get_transform(image_arr).apply_image(image_arr)
        image_arr = noise_maker(image_arr)
        image_arr = torch.as_tensor(image_arr.astype("float32").transpose(2, 0, 1))

        with torch.no_grad():
            denoised_img = ((image_arr - 0.5) / 0.5).unsqueeze(0)
            denoised_img = self.model.N2VNet(denoised_img.to(self.device))
            denoised_img = (((denoised_img * 0.5) + 0.5).clamp(0.0, 1.0) * 255)[0].permute(1, 2, 0).detach().cpu().numpy()
            denoised_img = cv2.resize(denoised_img, dsize=(d_h, d_w))
        
        cv2.imwrite(save_path, denoised_img)
    
    def inference_on_single_image(self, 
                                  image_file: str, 
                                  save_dir: str, 
                                  image_scale: float = 1.0,
                                  grid_split: bool = False,
                                  split_size: int = None):
        
        seg_path = os.path.join(save_dir, 'Segmentation')
        mask_path = os.path.join(save_dir, 'BinaryMask')
        denoised_path = os.path.join(save_dir, 'Denoised')
        traffic_path = os.path.join(save_dir, 'Traffic')
        
        for p in [seg_path, mask_path, denoised_path, traffic_path]:
            os.makedirs(p, exist_ok=True)
        
        if not grid_split:
            img_arr = load_image(image_file)
            pred, _, denoised_img = self(img_arr)
            
            # Draw predicted instances on 'img_arr'
            v = Visualizer(img_arr, metadata=self.metadata, scale=image_scale, instance_mode=ColorMode.IMAGE_BW)
            out = v.draw_instance_predictions(pred['instances'].to('cpu'))
            out = out.get_image()[:, :, ::-1]
        
            # Extract binary mask from prediction
            instance_mask = self._extract_binary_mask(pred['instances'])
            traffic = self._overlay_mask_on_image(img_arr[:, :, ::-1], instance_mask)
            
        else:
            assert split_size is not None, "When 'grid_split' is True, 'split_size' cannot be None."
            
            img_arr = load_image(image_file)
            ori_img_h, ori_img_w = img_arr.shape[:2]
            
            # Split large image into smaller patches
            slices, pos = split_image_into_slices(image_file, split_size)
            
            instance_slices, mask_slices = [], []
            for slice, p in zip(slices, pos):
                
                slice_pred, input_meta = self._base_call(slice)
                origin_shape = (input_meta['height'], input_meta['width'])
                
                v = Visualizer(slice, metadata=self.metadata, scale=1.0, instance_mode=ColorMode.IMAGE_BW)
                out = v.draw_instance_predictions(slice_pred['instances'].to('cpu'))
                instance_slices.append(cv2.resize(out.get_image()[:, :, ::-1], origin_shape))
                
                mask_slice = self._extract_binary_mask(slice_pred['instances'])
                mask_slices.append(cv2.resize(mask_slice, origin_shape))
            
            # Merge the patches and trim the zero-filled edges.
            out = merge_slices(instance_slices, pos)[:ori_img_h, :ori_img_w]
            instance_mask = merge_slices(mask_slices, pos)[:ori_img_h, :ori_img_w]
            traffic = self._overlay_mask_on_image(img_arr, instance_mask)
        
        # Save results
        cv2.imwrite(os.path.join(seg_path, os.path.basename(image_file)), out)
        cv2.imwrite(os.path.join(mask_path, os.path.basename(image_file)), instance_mask)
        cv2.imwrite(os.path.join(denoised_path, os.path.basename(image_file)), denoised_img)
        cv2.imwrite(os.path.join(traffic_path, os.path.basename(image_file)), traffic)
        
        print(f"* Inference finished. Result files are saved to '{save_dir}'.")
        