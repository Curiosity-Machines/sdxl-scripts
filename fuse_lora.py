import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image

model_id = "segmind/Segmind-Vega"
adapter_id = "gfodor/bigp1xart-vega-detailed-256x256"

pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")

pipe.load_lora_weights(adapter_id)
pipe.fuse_lora()
pipe.save_pretrained("Segmind-Vega-Pix")
