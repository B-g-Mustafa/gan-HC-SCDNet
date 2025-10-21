import torch
from torchvision import transforms
from PIL import Image
from beta_vae import BetaVAE # your trained model class
import os

# ----- Load model -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = '/home/msai/birul001/gan-project/gan-HC-SCDNet/b-vae/model/checkpoints/beta_vae_epoch30.pth'
ckpt = torch.load(ckpt_path, map_location=device)

model = BetaVAE(img_size=256, style_dim=128, content_dim=128).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()

# ----- Transforms -----
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ----- Load images -----
content_path = "/home/msai/birul001/gan-project/gan-HC-SCDNet/b-vae/infer/content.jpg"  # your real photo
style_path   = "/home/msai/birul001/gan-project/gan-HC-SCDNet/b-vae/infer/style.jpg"    # your chosen artwork

content_img = transform(Image.open(content_path).convert('RGB')).unsqueeze(0).to(device)
style_img   = transform(Image.open(style_path).convert('RGB')).unsqueeze(0).to(device)

# ----- Encode -----
mu_s, lv_s, _, _ = model._forward_stats(style_img)
_, _, mu_c, lv_c = model._forward_stats(content_img)

# ----- Sample latents -----
z_s = model.reparam(mu_s, lv_s)
z_c = model.reparam(mu_c, lv_c)

# # ----- Decode (mix style + content) -----
# with torch.no_grad():
#     output = model.decode_from_latent(z_s, z_c)

# # ----- Save output -----
# os.makedirs('results', exist_ok=True)
# out_img = transforms.ToPILImage()(output.squeeze().cpu().clamp(0, 1))
# out_img.save('results/stylized_result.png')
# print("✅ Saved stylized image to results/stylized_result.png")

z_style_zero = torch.zeros_like(z_s)
output_content = model.decode_from_latent(z_style_zero, z_c)

z_content_zero = torch.zeros_like(z_c)
output_style = model.decode_from_latent(z_s, z_content_zero)


def save_tensor_image(tensor, path):
    img = transforms.ToPILImage()(tensor.squeeze().cpu().clamp(0, 1))
    img.save(path)

os.makedirs("results", exist_ok=True)
save_tensor_image(output_content, "results/content_only.png")
save_tensor_image(output_style, "results/style_only.png")
print("✅ Saved content_only.png and style_only.png")

