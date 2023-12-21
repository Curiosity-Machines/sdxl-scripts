To get vega working:
- First onnx convert it as fp32 using optimum
- Then, convert the unet using the tune script, to fp16
- Then, grab the vae from here https://huggingface.co/madebyollin/sdxl-vae-fp16-fix, convert with optimum (not fp16), then convert to fp16 with tune script
