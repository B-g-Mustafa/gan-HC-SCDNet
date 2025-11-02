import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import IPAdapterImageEncoder, load_image
from diffusers.utils import load_image as load_img_utils

# 1. Load SD3 (or SDXL) base pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3",  # or SDXL path
    torch_dtype=torch.float16
).to("cuda")

# 2. Load LoRA weights (style fine-tuned)
pipe.load_attn_procs("path/to/lora_weights")  # directory containing LoRA adapter weights

# 3. Load IP-Adapter (for style image guidance)
pipe.load_ip_adapter(
    "path/to/ip-adapter.bin",  # Path to IP-Adapter weights
    image_encoder=IPAdapterImageEncoder.from_pretrained("h94/IP-Adapter")
)

# 4. (Optional) Load ControlNet for structure control (e.g., pose, edge map)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose"  # or depth/canny/other structure
).to("cuda")
pipe.controlnet = controlnet

# 5. Prepare input images
base_img = load_img_utils("path/to/base_image.png", size=512)
style_img = load_img_utils("path/to/style_image.png", size=512)

# 6. Run generation with style and structure control
generated = pipe(
    prompt="A beautiful scene in the style...",  # can be a short description or empty
    image_prompt=style_img,                      # for IP-Adapter
    input_image=base_img,                        # for ControlNet structure guidance
    num_inference_steps=30,
    ip_adapter_scale=0.8                         # controls style weight from the IP-Adapter
).images[0]

generated.save("stylized_output.png")
