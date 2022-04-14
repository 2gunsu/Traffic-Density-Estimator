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

from engine import BasePredictor
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
    

class MaskRCNNPredictor(BasePredictor):
    def __call__(self, image_arr: np.ndarray):
        return self._base_call(image_arr)[0]

    def inference_on_single_image(self, image_file: str, save_path: str, image_scale: float = 2):
        img_arr = cv2.imread(image_file)[:, :, ::-1]
        pred = self(img_arr)
        
        v = Visualizer(img_arr, metadata=self.metadata, scale=image_scale, instance_mode=ColorMode.IMAGE)
        out = v.draw_instance_predictions(pred['instances'].to('cpu'))
        out = out.get_image()[:, :, ::-1]
        
        cv2.imwrite(save_path, out)