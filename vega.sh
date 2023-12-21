#!/usr/bin/env bash
git clone https://huggingface.co/segmind/Segmind-Vega
optimum-cli export onnx --fp16 --model ../Segmind-Vega --task stable-diffusion-xl --device cuda vega_fp16
python tune-vega.py
