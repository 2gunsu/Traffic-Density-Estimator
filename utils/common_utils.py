import os
import sys
import json
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from yacs.config import CfgNode
from detectron2.model_zoo import model_zoo
from detectron2.data.datasets import register_coco_instances

from utils.default_config import _C


def load_cfg_arch(arch_name: str, cfg: CfgNode = None) -> CfgNode:
    _ARCH_DICT = {'R50-C4': 'R_50_C4_3x.yaml',
                  'R50-DC5': 'R_50_DC5_3x.yaml',
                  'R50-FPN': 'R_50_FPN_3x.yaml',
                  'R101-C4': 'R_101_C4_3x.yaml',
                  'R101-DC5': 'R_101_DC5_3x.yaml',
                  'R101-FPN': 'R_101_FPN_3x.yaml',
                  'X101-FPN': 'X_101_32x8d_FPN_3x.yaml'}
    
    _PATH_PREFIX = "COCO-InstanceSegmentation/mask_rcnn_"
    
    if cfg is None:
        cfg = _C.clone()
    
    assert arch_name in _ARCH_DICT.keys(), \
        f"Argument 'arch_name' must be one in {list(_ARCH_DICT.keys())}."
    
    config_path = (_PATH_PREFIX + _ARCH_DICT[arch_name])
    
    cfg.merge_from_file(model_zoo.get_config_file(config_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
    return cfg


def register_dataset(data_root: str, register_name: str):
    image_root = os.path.join(data_root, 'Image')
    annot_file = os.path.join(data_root, 'Label.json')
    
    register_coco_instances(name=register_name, metadata={}, json_file=annot_file, image_root=image_root)


def export_config(cfg: CfgNode, save_path: str):
    with open(save_path, "w") as f:
        yaml.dump(yaml.safe_load(cfg.dump()), f)
        
        
def get_num_classes(annot_file: str) -> int:
    with open(annot_file, 'r') as json_file:
        data = json.load(json_file)
    return len(data['categories'])
