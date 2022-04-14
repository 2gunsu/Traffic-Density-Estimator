import os
import cv2
import sys
import torch
import numpy as np
import detectron2.data.transforms as T

from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.structures import Instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer, ColorMode

from utils.default_config import _C


class BasePredictor:
    def __init__(self, cfg_file: str, weight_file: str, score_thres: float = 0.60):
        
        # Load Config
        cfg = _C.clone()
        cfg.set_new_allowed(True)
        cfg.merge_from_file(cfg_file)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thres
        
        self.cfg = cfg.clone()
        self.device = cfg.MODEL.DEVICE
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
        raise NotImplementedError
        
    def _base_call(self, image_arr: np.ndarray):
        with torch.no_grad():
            height, width = image_arr.shape[:2]
            image = self.aug.get_transform(image_arr).apply_image(image_arr)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions, inputs
    
    def inference_on_single_image(self, image_file: str, save_path: str, image_scale: float = 2.0):
        raise NotImplementedError
        
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