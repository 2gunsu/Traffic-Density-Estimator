import os
import warnings
import argparse

from datetime import datetime
from detectron2.data import DatasetCatalog, MetadataCatalog

from model import ModifiedRCNN
from engine import MaskRCNNTrainer, DMRCNNTrainer
from utils.common_utils import load_cfg_arch, register_dataset, get_num_classes, export_config

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description="Trainer")

parser.add_argument('--train_path', type=str, help="Directory of training data")
parser.add_argument('--val_path', type=str, default="", help="Directory of validation data")

parser.add_argument('--gpu_id', type=int, default=0, help="The index of the GPU to be used for training. (Only single GPU available)")
parser.add_argument('--output_dir', type=str, default="", help="Output directory where training results are saved to")

parser.add_argument('--input_size', type=int, default=800, help="Determinte the size of the image to be used for training")
parser.add_argument('--noise_type', type=str, default='none', help="What kind of noise to be added",
                    choices=['none', 'gaussian', 'speckle', 'salt&pepper', 'mix'])
parser.add_argument('--noise_params', nargs="+", default=[], help="Parameters for controlling the noise")

parser.add_argument('--backbone_arch', type=str, default='X101-FPN', help="Architecture of backbone network",
                    choices=['R50-C4', 'R50-DC5', 'R50-FPN', 'R101-C4', 'R101-DC5', 'R101-FPN', 'X101-FPN'])
parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
parser.add_argument('--base_lr', type=float, default=2.0e-03, help="Learning rate")
parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
parser.add_argument('--num_workers', type=int, default=8, help="Number of workers")

parser.add_argument('--save_period', type=int, default=5000, help="Step interval to save training results")
parser.add_argument('--val_period', type=int, default=2000, help="Step interval to perform validation")

# Arguments for Noise2Void
parser.add_argument('--n2v', action='store_true', help="Whether to use Noise2Void(N2V)")

parser.add_argument('--patch_size', type=int, default=64, help="Size of patch used to train Noise2Void")
parser.add_argument('--patch_nums', type=int, default=8, help="Number of patches per image")
parser.add_argument('--unet_base_ch', type=int, default=32, help="Number of base channels in U-Net for Noise2Void")
parser.add_argument('--win_size', type=int, default=5, help="Size of window used to generate Noise2Void data")
parser.add_argument('--spot_ratio', type=float, default=0.3, help="Ratio of blind spot")
parser.add_argument('--n2v_lr', type=float, default=2.5e-04, help="Learning rate of Noise2Void")


if __name__ == "__main__":
    
    args = parser.parse_args()
    
    use_n2v = args.n2v
    trainer_cls = DMRCNNTrainer if use_n2v else MaskRCNNTrainer
    
    # Get Default Configuration
    cfg = load_cfg_arch(args.backbone_arch)
    
    
    # Register Datasets
    register_dataset(data_root=args.train_path, register_name='train')
    do_validation = (len(args.val_path) != 0)
    if do_validation:
        register_dataset(data_root=args.val_path, register_name='val')
        
    # Create Output Directory
    if (len(args.output_dir) != 0):
        output_dir = args.output_dir
    else:
        # Convert '2022-04-13 21:49:20.548212' ---> '20220413_215012'
        time_string = str(datetime.now())[:19].replace('-', '').replace(':', '')
        time_string = '_'.join(time_string.split())
        output_dir = os.path.join('checkpoint', time_string)
    os.makedirs(output_dir, exist_ok=True)
    
    
    # Set Parameters in Config
    cfg.OUTPUT_DIR = output_dir
    
    cfg.DATASETS.TRAIN = ('train', )
    cfg.DATASETS.TEST = ('val', ) if do_validation else ('train', )
    
    cfg.MODEL.DEVICE = f"cuda:{str(args.gpu_id)}"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = get_num_classes(os.path.join(args.train_path, 'Label.json'))
    
    cfg.SOLVER.MAX_ITER = (len(DatasetCatalog.get('train')) // args.batch_size) * args.epochs
    cfg.SOLVER.STEPS = (int(cfg.SOLVER.MAX_ITER * 0.75), )
    cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.MAX_ITER * 0.10)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / cfg.SOLVER.WARMUP_ITERS
    
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.CHECKPOINT_PERIOD = args.save_period
    
    cfg.TEST.EVAL_PERIOD = args.val_period
    cfg.TEST.THING_CLASSES = MetadataCatalog.get('train').get("thing_classes", None)
    
    # Add Additional Params for Noise2Void
    
    if use_n2v:
        cfg.MODEL.META_ARCHITECTURE = "ModifiedRCNN"
    
    cfg.MODEL.N2V.USE = use_n2v
    cfg.MODEL.N2V.PATCH_SIZE = args.patch_size
    cfg.MODEL.N2V.PATCH_PER_IMG = args.patch_nums
    cfg.MODEL.N2V.INITIAL_CH = args.unet_base_ch
    cfg.MODEL.N2V.WINDOW_SIZE = args.win_size
    cfg.MODEL.N2V.BLIND_SPOT_RATIO = args.spot_ratio
    cfg.MODEL.N2V.LR = args.n2v_lr
    
    
    # Start Training
    trainer = trainer_cls(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    
    # Export Config File
    cfg.MODEL.WEIGHTS = os.path.join(output_dir, 'model_final.pth')
    export_config(cfg, save_path=os.path.join(output_dir, 'config.yaml'))
    
    