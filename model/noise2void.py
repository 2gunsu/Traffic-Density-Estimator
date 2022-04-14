import os
import sys
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from typing import List, Dict
from yacs.config import CfgNode

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model import UNet
from utils.transforms import Normalizer, Denormalizer, RandomCrop


class Noise2Void(nn.Module):
    def __init__(self, cfg: CfgNode):
        
        super().__init__()
        
        self.cfg = cfg
        self.model = UNet(self.cfg.INPUT.IN_CHANNELS, self.cfg.MODEL.N2V.INITIAL_CH)
        self.device = self.cfg.MODEL.DEVICE
        
        self.normalizer = Normalizer(mean=0.5, std=0.5)
        self.denormalizer = Denormalizer(mean=0.5, std=0.5)

    def forward(self, batched_inputs: List[Dict]):
        if not self.training:
            return self.inference(batched_inputs)

        input_tensors = torch.cat([x['image'].unsqueeze(dim=0) for x in batched_inputs], dim=0)
        n2v_input, n2v_mask, n2v_label = self._generate_noise2void_data(input_tensors,
                                                                        patch_per_image=self.cfg.MODEL.N2V.PATCH_PER_IMG,
                                                                        patch_size=self.cfg.MODEL.N2V.PATCH_SIZE,
                                                                        size_window=self.cfg.MODEL.N2V.WINDOW_SIZE,
                                                                        blind_spot_ratio=self.cfg.MODEL.N2V.BLIND_SPOT_RATIO,
                                                                        device=self.cfg.MODEL.DEVICE)

        n2v_output = self.model(n2v_input)
        n2v_loss = F.l1_loss(n2v_output * (1 - n2v_mask), n2v_label * (1 - n2v_mask))
        return n2v_loss

    def inference(self, batched_inputs: List[Dict]):
        assert not self.training, "Model is currently 'train' mode."

        image = self.normalizer(torch.cat([x['image'].unsqueeze(0) for x in batched_inputs]))

        denoised_image = self.model(image)
        denoised_image = self.denormalizer(denoised_image).clamp(min=0.0, max=1.0)
        return denoised_image

    def _generate_noise2void_data(self,
                                  label_tensor: torch.Tensor,
                                  patch_per_image: int = 8,
                                  patch_size: int = 64,
                                  size_window: int = 5,
                                  blind_spot_ratio: float = 0.3,
                                  device: str = 'cuda'):

        output_list, mask_list, label_list = [], [], []
        size_window = (size_window, size_window)

        for label in label_tensor:
            label_array = label.permute(1, 2, 0).detach().cpu().numpy()

            for _ in range(patch_per_image):
                cropped_label = RandomCrop(output_size=patch_size)(label_array)

                num_sample = int((patch_size ** 2) * blind_spot_ratio)
                mask = np.ones((patch_size, patch_size, 3))
                output = copy.deepcopy(cropped_label)

                for ich in range(3):
                    idy_msk = np.random.randint(0, patch_size, num_sample)
                    idx_msk = np.random.randint(0, patch_size, num_sample)

                    idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2,
                                                  size_window[0] // 2 + size_window[0] % 2,
                                                  num_sample)
                    idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2,
                                                  size_window[1] // 2 + size_window[1] % 2,
                                                  num_sample)

                    idy_msk_neigh = idy_msk + idy_neigh
                    idx_msk_neigh = idx_msk + idx_neigh

                    idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * patch_size - (
                            idy_msk_neigh >= patch_size) * patch_size
                    idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * patch_size - (
                            idx_msk_neigh >= patch_size) * patch_size

                    id_msk = (idy_msk, idx_msk, ich)
                    id_msk_neigh = (idy_msk_neigh, idx_msk_neigh, ich)

                    output[id_msk] = cropped_label[id_msk_neigh]
                    mask[id_msk] = 0.0

                output_list.append(self._numpy_to_tensor(output))
                mask_list.append(self._numpy_to_tensor(mask))
                label_list.append(self._numpy_to_tensor(cropped_label))

        # Concat
        output = torch.cat(output_list, dim=0).to(device)
        mask = torch.cat(mask_list, dim=0).to(device)
        label = torch.cat(label_list, dim=0).to(device)
        
        # Normalize
        n2v_label = self.normalizer(label).type(torch.FloatTensor).to(self.device)
        n2v_input = self.normalizer(output).type(torch.FloatTensor).to(self.device)
        n2v_mask = self.normalizer(mask).type(torch.FloatTensor).to(self.device)
        
        return n2v_label, n2v_input, n2v_mask
    
    def _numpy_to_tensor(self, numpy_arr: np.ndarray):
        assert numpy_arr.ndim == 3
        return torch.from_numpy(numpy_arr).permute(2, 0, 1).unsqueeze(0)
