import os
import warnings
import argparse

from engine.default_engine import MaskRCNNPredictor

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description="Pure Mask R-CNN")

parser.add_argument('--config_file', type=str, required=True, help="Path of config file (.yaml)")
parser.add_argument('--weight_file', type=str, required=True, help="Path of weight file (.pth)")
parser.add_argument('--conf_score', type=float, default=0.6, help="Confidence threshold for inference")

# Inference for Single Image File
parser.add_argument('--image_file', type=str, default='', help="Path of single image file")
parser.add_argument('--save_path', type=str, default='')

# Inference for Multiple Image Files
parser.add_argument('--image_dir', type=str, default='', help="Directory which contains multiple image files")
parser.add_argument('--save_dir', type=str, default='')

parser.add_argument('--gpu_id', type=int, default=0, help="The index of the GPU to be used for inference (Only Single GPU Available)")

parser.add_argument('--input_size', type=int, default=800, help="Determinte the size of the image to be used for inference")
parser.add_argument('--output_scale', type=float, default=2.0)



if __name__ == "__main__":
    
    args = parser.parse_args()
    
    # Initialize Predictor
    predictor = MaskRCNNPredictor(args.config_file, args.weight_file, args.conf_score)
    
    if len(args.image_file) > 0:
        assert os.path.isfile(args.image_file), "Cannot find 'image_file' you entered."
        assert len(args.save_path) > 0, "In single image mode, you must enter parameter 'save_path'."
        predictor.inference_on_single_image(args.image_file, args.save_path, args.output_scale)
        
    elif len(args.image_dir) > 0:
        assert os.path.isdir(args.image_dir), "Cannot find 'image_dir' you entered."
        assert len(args.save_dir) > 0, "In multi image mode, you must enter parameter 'save_dir'."
        predictor.inference_on_multi_images(args.image_dir, args.save_dir, args.output_scale)
        
    