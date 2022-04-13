import os
import sys
import torch
import numpy as np

from typing import List
from torchvision.transforms import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class Identity:
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image


class AddGaussianNoise:
    def __init__(self, std: List[int], drange: float = 255.):
        assert drange in [1.0, 255.0], "Argument 'drange' must be one in [0.0, 255.0]."
        
        self.std = std
        self.drange = drange

    def __call__(self, image: np.ndarray) -> np.ndarray:
        std = np.random.choice(self.std, 1)[0]
        image = image.astype('float32') + (np.random.randn(*image.shape) * std)
        return np.clip(image, a_min=0.0, a_max=self.drange)


class AddSpeckleNoise:
    def __init__(self, mean: float, std: List[float]):
        self.mean = mean
        self.std = std

    def __call__(self, image: np.ndarray) -> np.ndarray:
        std = np.random.choice(self.std, 1)[0]
        noise = np.random.normal(self.mean, std, image.shape)
        image = image + image * noise
        return image


class AddSaltPepperNoise:
    def __init__(self, amount: List[float], drange: float = 255.):
        assert drange in [1.0, 255.0], "Argument 'drange' must be one in [0.0, 255.0]."
        
        self.amount = amount
        self.drange = drange

    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert image.ndim >= 2, "Dimension of 'image' must be equal to or larger than 2."

        amount = np.random.choice(self.amount, 1)[0]
        flipped = np.random.choice([True, False], size=image.shape, p=[amount, 1 - amount])
        salted = np.random.choice([True, False], size=image.shape, p=[0.5, 0.5])
        peppered = ~salted

        image[flipped * salted] = self.drange
        image[flipped * peppered] = 0
        return image


class AddRandomNoise:
    def __init__(self, drange: float = 255.):
        assert drange in [1.0, 255.0], "Argument 'drange' must be one in [0.0, 255.0]."
        
        self.drange = drange
        self.noise_list = [
            AddGaussianNoise([15, 30, 50], self.drange),
            AddSpeckleNoise(0.0, [0.1, 0.2]),
            AddSaltPepperNoise([0.05, 0.10, 0.15], self.drange)]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        random_idx = np.random.choice(len(self.noise_list), 1)[0]
        return self.noise_list[random_idx](image)


class RandomCrop:
    def __init__(self, output_size: int):
        self.output_size = (output_size, output_size)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        new_height, new_width = self.output_size

        top = np.random.randint(0, height - new_height)
        left = np.random.randint(0, width - new_width)

        id_y = np.arange(top, top + new_height, 1)[:, np.newaxis].astype(np.int32)
        id_x = np.arange(left, left + new_width, 1).astype(np.int32)
        return image[id_y, id_x]


class Normalizer:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3:
            return transforms.Normalize(mean=self.mean, std=self.std)(tensor)
        elif tensor.dim() == 4:
            return torch.cat([transforms.Normalize(mean=self.mean, std=self.std)(t).unsqueeze(0) for t in tensor], dim=0)
        else:
            raise Exception("Dimension of 'tensor' must be equal to or larger than 3.")


class Denormalizer:
    def __init__(self, mean: float, std: float):
        
        self.mean = mean
        self.std = std

        if isinstance(self.mean, float):
            self.mean = self._match_dimension(3, self.mean)
        if isinstance(self.std, float):
            self.std = self._match_dimension(3, self.std)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3:
            return transforms.Normalize(mean=[(-m / s) for m, s in zip(self.mean, self.std)], std=[(1 / s) for s in self.std])(tensor)
        elif tensor.dim() == 4:
            return torch.cat([transforms.Normalize(mean=[(-m / s) for m, s in zip(self.mean, self.std)],
                                                   std=[(1 / s) for s in self.std])(t).unsqueeze(0) for t in tensor], dim=0)
        else:
            raise Exception("Dimension of 'tensor' must be equal to or larger than 3.")

    def _match_dimension(self, desired_dim: int, input_param: float):
        return [input_param for _ in range(desired_dim)]