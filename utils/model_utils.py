import os
import sys
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def transfer_weight(src_model: nn.Module, dst_model: nn.Module):
    src_dict = src_model.state_dict()
    dst_dict = dst_model.state_dict()
    assert len(src_dict.keys()) == len(dst_dict.keys()), \
        "Source module and Destination module seems different."

    for src_key, dst_key in zip(src_dict.keys(), dst_dict.keys()):
        dst_dict[dst_key] = src_dict[src_key]
    dst_model.load_state_dict(dst_dict)
