import os
import cv2
import sys
import copy
import torch
import numpy as np
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils

from itertools import product
from typing import Union, Tuple, List
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.common_utils import register_dataset
from utils.transforms import Identity, AddGaussianNoise, AddSaltPepperNoise, AddSpeckleNoise, AddRandomNoise


_SUPPORT_NOISE = ['none', 'gaussian', 'speckle', 'salt&pepper', 'mix']


def get_noise_adder(noise_type: str, 
                    noise_param: List[Union[int, float]] = [], 
                    drange: float = 255.):
    
    assert noise_type in _SUPPORT_NOISE, \
        f"Argument 'noise_type' must be one in {noise_type}."

    if noise_type == 'none':
        return Identity()

    elif noise_type == 'gaussian':
        assert isinstance(noise_param, list), \
            "When in 'gaussian' mode, 'noise_param' should be list. " \
            "(i.e. [15] or [15, 30, 50])"
        return AddGaussianNoise(std=noise_param, drange=drange)

    elif noise_type == 'speckle':
        assert isinstance(noise_param, list), \
            "When in 'speckle' mode, 'noise_param' should be list. " \
            "(i.e. [0.2] or [0.1, 0.2, 0.3])"
        return AddSpeckleNoise(mean=0.0, std=noise_param)

    elif noise_type == 'salt&pepper':
        assert isinstance(noise_param, list), \
            "When in 'salt&pepper' mode, 'noise_param' should be list. " \
            "(i.e. [0.05] or [0.05, 0.10, 0.15])"
        return AddSaltPepperNoise(amount=noise_param, drange=drange)

    elif noise_type == 'mix':
        return AddRandomNoise(drange=drange)


def build_mapper(resize: Union[Tuple[int, int], int],
                 noise_type: str = 'none',
                 noise_param: List[Union[int, float]] = [],
                 drange: float = 255.,
                 use_n2v: bool = False):

    assert drange in [1.0, 255.0], "Argument 'drange' must be one in [0.0, 255.0]."

    if isinstance(resize, int):
        resize = (resize, resize)
        
    noise_maker = get_noise_adder(noise_type, noise_param)
    scaler = 1.0 if not use_n2v else 255.

    def mapper(dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict['file_name'], format='BGR')

        image, transforms = T.apply_transform_gens([T.Resize(shape=resize)], image)
        image = noise_maker(image)

        dataset_dict['image'] = torch.as_tensor(image.transpose(2, 0, 1).astype('float32')) / scaler
        annos = [utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                 for obj in dataset_dict.pop("annotations")
                 if obj.get("iscrowd", 0) == 0]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict['instances'] = utils.filter_empty_instances(instances)
        return dataset_dict
    return mapper


def visualize_coco_dataset(data_root: str, save_path: str):
    register_dataset(data_root, 'vis_dataset')
    
    meta = MetadataCatalog.get('vis_dataset')
    dataset_dicts = DatasetCatalog.get('vis_dataset')
    
    os.makedirs(save_path, exist_ok=True)
    
    print(f"* Visualizing COCO Format Dataset...")
    print("=" * 120) 
    for idx, d in enumerate(dataset_dicts):
        img = cv2.imread(d['file_name'])[:, :, ::-1]
        img_copy = img.copy()
        
        visualizer = Visualizer(img_copy, metadata=meta, scale=1.0)
        vis = visualizer.draw_dataset_dict(d).get_image()[:, :, ::-1]
        
        print(f"[{idx + 1:5d} / {len(dataset_dicts):5d}] Saved to '{os.path.join(save_path, os.path.basename(d['file_name']))}'")
        cv2.imwrite(os.path.join(save_path, os.path.basename(d['file_name'])), vis)
    print("=" * 120) 


def load_image(image_file: str, load_as_tensor: bool = False, verbose: bool = False) -> Union[np.ndarray, torch.Tensor]:
    # Image File --> np.ndarray (0 ~ 255)
    img_arr = cv2.imread(image_file)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    
    if verbose:
        print(f"* Image '{image_file}' is loaded. (Shape: {img_arr.shape})")
    
    # np.ndarray (0 ~ 255) --> torch.Tensor (0.0 ~ 1.0)
    if load_as_tensor:
        img_arr = torch.from_numpy(img_arr).permute(2, 0, 1) / 255.
    return img_arr


def split_image_into_slices(image_file: str, 
                            slice_hw: Union[int, Tuple[int, int]], 
                            allow_overlap: bool = False,
                            save_path: str = None):
    
    img_arr = load_image(image_file)
    img_h, img_w = img_arr.shape[:2]
    
    if isinstance(slice_hw, int):
        slice_hw = (slice_hw, slice_hw)
    slice_h, slice_w = slice_hw
    
    num_h, num_w = (img_h // slice_h), (img_w // slice_w)
    
    if (img_h % slice_h > 0):
        num_h += 1
        
    if (img_w % slice_w > 0):
        num_w += 1
        
    o_h, o_w = 0, 0
    if allow_overlap:
        o_h = ((slice_h * num_h) - img_h) // (num_h - 1)
        o_w = ((slice_w * num_w) - img_w) // (num_w - 1)
        
    # If overlap is not allowed, it expands by filling in the value of the insufficient space with 0.
    else:
        expanded_img = np.zeros((slice_h * num_h, slice_w * num_w, 3), dtype=np.uint8)
        expanded_img[:img_h, :img_w, :] = img_arr
        img_arr = expanded_img
        
    slices = []
    pos = []
    
    for h_idx, w_idx in product(range(0, num_h), range(0, num_w)):
        
        lt_coor = np.array([w_idx * (slice_w - o_w), (h_idx * (slice_h - o_h))])
        rb_coor = lt_coor + np.array([slice_w, slice_h])
        
        slice = img_arr[lt_coor[1]: rb_coor[1], lt_coor[0]: rb_coor[0], :]
        slices.append(slice)
        pos.append([h_idx, w_idx])
    
    # If 'save_path' is None, return the results.
    if not save_path:
        return slices, pos
    
    # If 'save_path' is not None, all slices are saved as image files.
    os.makedirs(save_path, exist_ok=True)
    for slice, p in zip(slices, pos):
        
        f_name, f_ext = os.path.splitext(os.path.basename(image_file))
        file_name = f"{f_name}_C{str(p[0]).zfill(2)}_R{str(p[1]).zfill(2)}_{f_ext}"
        
        slice = cv2.cvtColor(slice, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path, file_name), slice)


def merge_slices(slices: List[np.ndarray], 
                 pos_info: List[List[int]], 
                 save_path: str = None):
    
    slice_h, slice_w = slices[0].shape[:2]
    is_rgb = True if (slices[0].ndim == 3) else False
    
    pos_info = np.array(pos_info)
    max_h, max_w = np.max(pos_info[:, 0]) + 1, np.max(pos_info[:, 1]) + 1
    
    image_h, image_w = (slice_h * max_h), (slice_w * max_w)
    merged_arr = np.zeros((image_h, image_w, 3), dtype=np.uint8) if is_rgb else np.zeros((image_h, image_w), dtype=np.uint8)

    for slice, pos in zip(slices, pos_info):
        
        lt_y = pos[0] * slice_h
        rb_y = lt_y + slice_h
        
        lt_x = pos[1] * slice_w
        rb_x = lt_x + slice_w
        
        merged_arr[lt_y: rb_y, lt_x: rb_x] = slice
        
    if not save_path:
        return merged_arr

    merged_arr = cv2.cvtColor(merged_arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, merged_arr)
