import os
import sys
import torch
import torch.nn as nn

from yacs.config import CfgNode
from typing import Tuple, List, Union, Dict
from torchvision.transforms import transforms

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList, Instances
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model import UNet
from utils.transforms import Normalizer, Denormalizer


# Noise2Void + R-CNN
@META_ARCH_REGISTRY.register()
class ModifiedRCNN(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        N2VNet: nn.Module,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):

        super().__init__()
        
        self.N2VNet = N2VNet                                # Noise2Void Network
        self.backbone = backbone                            # Backbone
        self.proposal_generator = proposal_generator        # RPN
        self.roi_heads = roi_heads                          # RoI Heads

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        
        self.n2v_norm = Normalizer(mean=0.5, std=0.5)
        self.n2v_denorm = Denormalizer(mean=0.5, std=0.5)
        self.rcnn_norm = Normalizer(mean=self.pixel_mean, std=self.pixel_std)

    @classmethod
    def from_config(cls, cfg: CfgNode):
        backbone = build_backbone(cfg)
        return {
            "N2VNet": UNet(cfg.INPUT.IN_CHANNELS, cfg.MODEL.N2V.INITIAL_CH),
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD}

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs: List[Dict]):
        if not self.training:
            return self.inference(batched_inputs)

        # Prepare Inputs
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        images = [x['image'] for x in batched_inputs]
        images_list = ImageList.from_tensors(tensors=images).to(self.device)

        # Noise2Void
        n2v_input = self.n2v_norm(images_list.tensor)
        n2v_input_list = ImageList.from_tensors(tensors=[tensor for tensor in n2v_input])
        n2v_output = self.N2VNet(n2v_input_list.tensor)
        n2v_output = self.n2v_denorm(n2v_output).clamp(0.0, 1.0)

        # Backbone of R-CNN
        backbone_input = n2v_output * 255.
        backbone_input = self.rcnn_norm(backbone_input)
        backbone_input_list = ImageList.from_tensors(tensors=[tensor for tensor in backbone_input])
        features = self.backbone(backbone_input_list.tensor)

        # RPN (Region Proposal Network)
        proposals, proposal_losses = self.proposal_generator(backbone_input_list, features, gt_instances)

        # RoI Heads
        _, detector_losses = self.roi_heads(backbone_input_list, features, proposals, gt_instances)

        # Update Losses
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs: List[Dict], do_postprocess: bool = True):
        assert not self.training, "Model is currently 'train' mode."

        # Prepare Inputs
        image = [x['image'] for x in batched_inputs]
        image_list = ImageList.from_tensors(tensors=image).to(self.device)
        
        # Noise2Void
        n2v_input = self.n2v_norm(image_list.tensor)
        n2v_input_list = ImageList.from_tensors(tensors=[tensor for tensor in n2v_input])
        denoised_image = self.N2VNet(n2v_input_list.tensor)
        denoised_image = self.n2v_denorm(denoised_image).clamp(0.0, 1.0)

        # Backbone of R-CNN
        denoised_image = denoised_image * 255.
        backbone_input = self.rcnn_norm(denoised_image)
        backbone_input_list = ImageList.from_tensors(tensors=[image for image in backbone_input])
        features = self.backbone(backbone_input_list.tensor)

        # RPN (Region Proposal Network)
        proposals, _ = self.proposal_generator(backbone_input_list, features, None)

        # RoI Heads
        results, _ = self.roi_heads(backbone_input_list, features, proposals, None)


        if do_postprocess:
            return ModifiedRCNN._postprocess(instances=results,
                                             batched_inputs=batched_inputs,
                                             image_sizes=image_list.image_sizes)
        else:
            return results

    @staticmethod
    def _postprocess(instances: Instances,
                     batched_inputs: List[Dict],
                     image_sizes: List[Tuple[int, int]]):
        processed_results = []
        for instance_per_img, single_img, image_size in zip(instances, batched_inputs, image_sizes):
            height = single_img.get("height", image_size[0])
            width = single_img.get("width", image_size[1])
            processed_results.append({"instances": detector_postprocess(instance_per_img, height, width)})
        return processed_results
