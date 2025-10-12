import gc
import math

# Simplified config for inference
config_text = """
DATA:
  NUM_FRAMES: 8
  SAMPLING_RATE: 8
  TEST_CROP_SIZE: 256
  INPUT_CHANNEL_NUM: [3]
RESNET:
  ZERO_INIT_FINAL_BN: True
  WIDTH_PER_GROUP: 64
  NUM_GROUPS: 1
  DEPTH: 50
  TRANS_FUNC: bottleneck_transform
  STRIDE_1X1: False
  NUM_BLOCK_TEMP_KERNEL: [[3], [4], [6], [3]]
NONLOCAL:
  LOCATION: [[[]], [[]], [[]], [[]]]
  GROUP: [[1], [1], [1], [1]]
  INSTANTIATION: softmax
BN:
  USE_PRECISE_STATS: True
  NUM_BATCHES_PRECISE: 200
MODEL:
  NUM_CLASSES: 1
  ARCH: i3d
  MODEL_NAME: ResNet
  DROPOUT_RATE: 0.1
  HEAD_ACT: sigmoid
TEST:
  ENABLE: True
  DATASET: kinetics
  BATCH_SIZE: 64
DATA_LOADER:
  NUM_WORKERS: 8
  PIN_MEMORY: True
NUM_GPUS: 8
NUM_SHARDS: 1
RNG_SEED: 0
OUTPUT_DIR: .
"""

# from .f3net import FAD_Head
from slowfast.models.video_model_builder import ResNet as ResNetOri
from slowfast.config.defaults import get_cfg
import torch
from torch import nn
from config_ftcn import config as my_cfg
from inspect import signature
# Removed TimeTransformer import - not used directly in this file
# Removed random import - not used in inference

my_cfg.init_with_yaml()
my_cfg.update_with_yaml("ftcn_tt.yaml")
my_cfg.freeze()



class CenterPatchPool(nn.Module):
    """Simplified patch pooling for inference - always use center patch"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # batch,channel,16,7x7
        b, c, t, h, w = x.shape
        x = x.reshape(b, c, t, h * w)
        idx = h * w // 2  # Always use center patch for inference
        x = x[..., idx]
        return x


def valid_idx(idx, h):
    i = idx // h
    j = idx % h
    if j == 0 or i == h - 1 or j == h - 1:
        return False
    else:
        return True


class CenterAvgPool(nn.Module):
    """Simplified average pooling for inference - use all valid patches"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # batch,channel,16,7x7
        b, c, t, h, w = x.shape
        x = x.reshape(b, c, t, h * w)
        candidates = list(range(h * w))
        candidates = [idx for idx in candidates if valid_idx(idx, h)]
        x = x[..., candidates].mean(-1)
        return x
    
        

# Removed duplicate TransformerHead class - using the one from temporal_transformer.py 


parameters = [parameter for parameter in signature(nn.Conv3d).parameters]

spatial_count = my_cfg.model.inco.spatial_count
keep_stride_count = my_cfg.model.inco.keep_stride_count


def temporal_only_conv(module, name, removed, stride_removed=0):
    """
    Recursively put desired batch norm in nn.module module.

    set module = net to start code.
    """
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        sub_module = getattr(module, attr_str)
        if type(sub_module) == nn.Conv3d:
            target_spatial_size = 1
            predefine_padding = {1: 0, 3: 1, 5: 2, 7: 3}
            kernel_size = list(sub_module.kernel_size)
            assert kernel_size[1] == kernel_size[2]
            stride = sub_module.stride
            extra = None
            if stride[1] == stride[2] == 2:
                stride_removed += 1
                if stride_removed > keep_stride_count:
                    stride = [1, 1, 1]
                    extra = nn.MaxPool3d((1, 2, 2))

            if kernel_size[1] == 1 and extra is None:
                continue
            padding = list(sub_module.padding)

            kernel_size[1] = kernel_size[2] = target_spatial_size
            padding[1] = padding[2] = predefine_padding[target_spatial_size]
            if 'device' in parameters:
                parameters.remove('device')
            if 'dtype' in parameters:
                parameters.remove('dtype')
            param_dict = {key: getattr(sub_module, key) for key in parameters}

            param_dict.update(kernel_size=kernel_size, padding=padding, stride=stride)

            conv = nn.Conv3d(**param_dict)

            new_module = conv

            removed += 1
            if removed > spatial_count:
                setattr(module, attr_str, new_module)
                if extra is not None:
                    if attr_str == "conv":
                        bn_str = "bn"
                    else:
                        bn_str = f"{attr_str}_bn"
                    bn_module = getattr(module, bn_str)
                    assert isinstance(bn_module, nn.BatchNorm3d)
                    new_bn_module = nn.Sequential(bn_module, extra)
                    setattr(module, bn_str, new_bn_module)
            else:
                print("keep spatial")
        elif type(sub_module) == nn.Dropout:
            new_module = nn.Dropout(p=0.5)
            setattr(module, attr_str, new_module)
        if my_cfg.model.inco.no_time_pool:
            if type(sub_module) == nn.MaxPool3d:
                kernel_size = list(sub_module.kernel_size)
                if kernel_size[0] == 2:
                    kernel_size[0] = 1
                    setattr(module, attr_str, nn.MaxPool3d(kernel_size))
            elif type(sub_module) == nn.AvgPool3d:
                kernel_size = list(sub_module.kernel_size)
                kernel_size[0] = 2 * kernel_size[0]
                setattr(module, attr_str, nn.AvgPool3d(kernel_size))

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    old_name = name
    for name, immediate_child_module in module.named_children():
        removed, stride_removed = temporal_only_conv(
            immediate_child_module, old_name + "." + name, removed, stride_removed
        )
    return removed, stride_removed


class I3D8x8(nn.Module):
    def __init__(self) -> None:
        super(I3D8x8, self).__init__()
        cfg = get_cfg()
        cfg.merge_from_str(config_text)
        cfg.NUM_GPUS = 1
        cfg.TEST.BATCH_SIZE = 1
        cfg.TRAIN.BATCH_SIZE = 1

        cfg.DATA.NUM_FRAMES = my_cfg.clip_size
        SOLVER = my_cfg.model.inco.SOLVER
        if SOLVER is not None:
            for key, val in SOLVER.to_dict().items():
                old_val = getattr(cfg.SOLVER, key)
                val = type(old_val)(val)
                setattr(cfg.SOLVER, key, val)

        if my_cfg.model.inco.i3d_routine:
            self.cfg = cfg
        self.resnet = ResNetOri(cfg)
        temporal_only_conv(self.resnet, "model", 0)

        stop_point = my_cfg.model.transformer.stop_point
        
        for i in [5, 4, 3]:
            if stop_point <= i:
                setattr(self.resnet, f"s{i}", nn.Identity())
                if stop_point == 3:
                    setattr(self.resnet, f"pathway0_pool", nn.Identity())
        gc.collect()
        torch.cuda.empty_cache()

    def forward(
            self,
            images, ft_features=None,
            freeze_backbone=False
    ):
        assert not freeze_backbone

        inputs = [images]
        pred = self.resnet(inputs, ft_features)
        return pred
