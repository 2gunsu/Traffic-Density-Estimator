import os
import sys
import copy
import torch
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils

from typing import Union, Tuple, List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

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