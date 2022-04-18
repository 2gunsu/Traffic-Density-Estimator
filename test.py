import os
import warnings
import argparse

from utils.default_config import _C
from engine import MaskRCNNPredictor, DMRCNNPredictor

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description="Predictor")

parser.add_argument('--config_file', type=str, required=True, help="Path of config file (.yaml)")
parser.add_argument('--weight_file', type=str, required=True, help="Path of weight file (.pth)")
parser.add_argument('--conf_score', type=float, default=0.6, help="Confidence threshold for inference")

parser.add_argument('--image_file', type=str, default='', help="Path of single image file")                         # Inference for Single Image File
parser.add_argument('--image_dir', type=str, default='', help="Directory which contains multiple image files")      # Inference for Multiple Image Files
parser.add_argument('--save_dir', type=str, default='')

parser.add_argument('--gpu_id', type=int, default=0, help="The index of the GPU to be used for inference (Only single GPU available)")

parser.add_argument('--input_size', type=int, default=800, help="Determinte the size of the image to be used for inference.")
parser.add_argument('--grid_split', action='store_true', help="Whether to proceed with inference by dividing the image into small patches")
parser.add_argument('--grid_size', type=int, default=None, help="Determine the size of patches")
parser.add_argument('--output_scale', type=float, default=1.0)



if __name__ == "__main__":
    
    args = parser.parse_args()
    
    # Check Config
    cfg = _C.clone()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    
    cfg.MODEL.DEVICE = f'cuda:{args.gpu_id}'
    cfg.INPUT.RESIZE = args.input_size
    
    
    # Initialize Predictor
    predictor_cls = DMRCNNPredictor if cfg.MODEL.N2V.USE else MaskRCNNPredictor
    predictor = predictor_cls(args.config_file, args.weight_file, args.conf_score)
    
    if len(args.image_file) > 0:
        assert os.path.isfile(args.image_file), "Cannot find 'image_file' you entered."
        predictor.inference_on_single_image(args.image_file, args.save_dir, args.output_scale, args.grid_split, args.grid_size)
        
    elif len(args.image_dir) > 0:
        assert os.path.isdir(args.image_dir), "Cannot find 'image_dir' you entered."
        predictor.inference_on_multi_images(args.image_dir, args.save_dir, args.output_scale, args.grid_split, args.grid_size)
        