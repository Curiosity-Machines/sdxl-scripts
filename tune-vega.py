import warnings
import argparse
import os
import shutil
from pathlib import Path
import json
import tempfile
from typing import Union, Optional, Tuple

import torch
from torch.onnx import export
import safetensors

import onnx
from diffusers.models import AutoencoderKL
from diffusers import (
    OnnxRuntimeModel,
    OnnxStableDiffusionPipeline,
    StableDiffusionXLPipeline,
    ControlNetModel,
    UNet2DConditionModel
)
from diffusers.models.attention_processor import AttnProcessor
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
from transformers.models.clip import CLIPTextModel

from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.optimizer import optimize_model

@torch.no_grad()
def tune_model(
    model_path: str,
    model_type: str,
    fp16: bool
):
    model_dir=os.path.dirname(model_path)
    
    # First we set our optimisation to the ORT Optimizer defaults for the provided type
    optimization_options = FusionOptions(model_type)
    # The ORT optimizer is designed for ORT GPU and CUDA
    # To make things work with ORT DirectML, we disable some options
    # The GroupNorm op has a very negative effect on VRAM and CPU use
    optimization_options.enable_group_norm = False
    # On by default in ORT optimizer, turned off as it causes performance issues
    optimization_options.enable_nhwc_conv = False
    # On by default in ORT optimizer, turned off because it has no effect
    optimization_options.enable_qordered_matmul = False
    optimization_options.enable_packed_qkv = False
    optimization_options.enable_packed_kv = False

    optimizer = optimize_model(
        input = model_path,
        model_type = model_type,
        opt_level = 0,
        optimization_options = optimization_options,
        use_gpu = False,
        only_onnxruntime = False
    )
    if fp16:
        optimizer.convert_float_to_float16(
        keep_io_types=True, disable_shape_infer=True, op_block_list=['RandomNormalLike','RandomNormal']
    )
    optimizer.topological_sort()
        
    shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    # collate external tensor files into one
    onnx.save_model(
        optimizer.model,
        model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx_data",
        convert_attribute=False,
    )        

unet_model_path = str(Path("vega_fp16/unet/model.onnx").absolute().as_posix())
vae_model_path = str(Path("vega_fp16/vae_decoder/model.onnx").absolute().as_posix())
tune_model(unet_model_path, "unet", True)
tune_model(vae_model_path, "vae", True)
