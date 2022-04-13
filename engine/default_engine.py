import os
import cv2
import sys
import torch
import numpy as np
import detectron2.data.transforms as T

from yacs.config import CfgNode
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer
from detectron2.structures import Instances
from detectron2.evaluation import COCOEvaluator
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import build_detection_test_loader, build_detection_train_loader, MetadataCatalog

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.default_config import _C
from utils.hook import LossEvalHook
from utils.data_utils import build_mapper


class MaskRCNNTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        mapper = build_mapper(resize=cfg.INPUT.RESIZE, 
                              noise_type=cfg.INPUT.NOISE_TYPE, 
                              noise_param=cfg.INPUT.NOISE_PARAM)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name: str):
        mapper = build_mapper(resize=cfg.INPUT.RESIZE, noise_type='none')
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg: CfgNode, dataset_name: str, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "evaluation")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                mapper=build_mapper(resize=self.cfg.INPUT.RESIZE, noise_type='none'))))
        return hooks


class MaskRCNNPredictor:
    def __init__(self, cfg_file: str, weight_file: str, score_thres: float = 0.60):
        
        # Load Config
        cfg = _C.clone()
        cfg.set_new_allowed(True)
        cfg.merge_from_file(cfg_file)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thres
        
        self.cfg = cfg.clone()
        self.score_thres = score_thres
        
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
            
            thing_classes = self.cfg.TEST.THING_CLASSES
            self.metadata.set(thing_classes=thing_classes)
            
        DetectionCheckpointer(self.model).load(weight_file)

        self.aug = T.Resize(cfg.INPUT.RESIZE)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, image_arr: np.ndarray):
        with torch.no_grad():
            height, width = image_arr.shape[:2]
            image = self.aug.get_transform(image_arr).apply_image(image_arr)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
    
    def inference_on_single_image(self, image_file: str, save_path: str, image_scale: float = 2.0):
        img_arr = cv2.imread(image_file)[:, :, ::-1]
        pred = self(img_arr)
        
        v = Visualizer(img_arr, metadata=self.metadata, scale=image_scale, instance_mode=ColorMode.IMAGE)
        out = v.draw_instance_predictions(pred['instances'].to('cpu'))
        out = out.get_image()[:, :, ::-1]
        
        cv2.imwrite(save_path, out)
        
    def inference_on_multi_images(self, image_dir: str, save_dir: str, image_scale: float = 2.0):
        os.makedirs(save_dir, exist_ok=True)
        image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
        
        for image_path in image_paths:
            self.inference_on_single_image(image_path, os.path.join(save_dir, os.path.basename(image_path)), image_scale)
            
    def _extract_binary_mask(self, instances: Instances) -> np.ndarray:
        scores = instances.scores.detach()
        score_mask = (scores >= self.score_thres)
        
        matched_binary_mask = (instances.pred_masks.detach())[score_mask]
        merged_mask = torch.clamp(matched_binary_mask.sum(dim=0), max=1.0)
        return (merged_mask.cpu().numpy() * 255.).astype(np.uint8)
