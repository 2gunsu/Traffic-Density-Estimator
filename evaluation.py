import os
import warnings
import argparse

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import inference_on_dataset, COCOEvaluator

from utils.default_config import _C
from utils.common_utils import register_dataset
from engine import MaskRCNNTrainer, DMRCNNTrainer

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description="Evaluator")

parser.add_argument('--eval_path', type=str, required=True, help="Directory of evaluation data")
parser.add_argument('--config_file', type=str, required=True, help="Path of config file (.yaml)")
parser.add_argument('--weight_file', type=str, required=True, help="Path of weight file (.pth)")
parser.add_argument('--output_dir', type=str, default='output', help="Output directory for evaluation results (Default: 'output')")

parser.add_argument('--gpu_id', type=int, default=0, help="The index of the GPU to be used for evaluation (Only Single GPU Available)")


if __name__ == "__main__":
    
    args = parser.parse_args()
    
    # Check Config
    cfg = _C.clone()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    
    cfg.MODEL.DEVICE = f'cuda:{args.gpu_id}'
    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(args.output_dir)
    
    
    # Build Evaluation Dataset
    register_dataset(args.eval_path, 'eval')
    cfg.DATASETS.TRAIN = ('eval', )
    cfg.DATASETS.TEST = ('eval', )
    
    
    # Build Trainer and Load Weight
    trainer_cls = DMRCNNTrainer if cfg.MODEL.N2V.USE else MaskRCNNTrainer
    attr = 'trainer.main_model' if cfg.MODEL.N2V.USE else 'trainer.model'
    
    trainer = trainer_cls(cfg)
    DetectionCheckpointer(eval(attr)).load(args.weight_file)
    
    
    # Build Evaluator and Evaluation Loader
    evaluator = COCOEvaluator('eval', ('bbox', 'segm', ), False, output_dir=None)
    eval_loader = build_detection_test_loader(cfg, 'eval')
    
    
    # Perform Evaluation
    result = inference_on_dataset(eval(attr), eval_loader, evaluator)