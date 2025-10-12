# Demo application and utilities (Fixed)
demo_code = '''
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
from hcscdnet_main import HCSCDNet
from typing import Tuple
import io
import base64

class HCSCDNetDemo:
    """
    Interactive demo application for HC-SCDNet
    Provides controllable style transfer interface
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load trained model
        self.model = HCSCDNet(device=self.device)
        if model_path and torch.cuda.is_available():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image for model input"""
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def postprocess_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert model output tensor to PIL image"""
        # Denormalize and clamp
        tensor = torch.clamp((tensor + 1) / 2, 0, 1)
        
        # Convert to PIL
        tensor_cpu = tensor.squeeze(0).cpu()
        image_np = tensor_cpu.permute(1, 2, 0).numpy()
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        
        return image_pil
    
    def generate_style_transfer(self,
                              content_image: Image.Image,
                              style_image: Image.Image,
                              style_strength: float = 1.0,
                              content_preservation: float = 1.0,
                              num_steps: int = 10) -> Image.Image:
        """Generate style transfer with controllable parameters"""
        with torch.no_grad():
            # Preprocess images
            content_tensor = self.preprocess_image(content_image)
            style_tensor = self.preprocess_image(style_image)
            
            # Generate
            outputs = self.model(
                content_tensor,
                style_tensor,
                style_control=style_strength,
                content_control=content_preservation,
                num_inference_steps=num_steps
            )
            
            # Postprocess
            generated_image = self.postprocess_image(outputs['generated_image'])
            
        return generated_image
    
    def create_control_grid(self,
                           content_image: Image.Image,
                           style_image: Image.Image) -> Image.Image:
        """Create a 3x3 grid showing different control combinations"""
        style_values = [0.2, 0.6, 1.0]
        content_values = [0.2, 0.6, 1.0]
        
        grid_images = []
        
        for style_val in style_values:
            row_images = []
            for content_val in content_values:
                generated = self.generate_style_transfer(
                    content_image, style_image,
                    style_strength=style_val,
                    content_preservation=content_val,
                    num_steps=8
                )
                row_images.append(np.array(generated))
            
            row_concat = np.concatenate(row_images, axis=1)
            grid_images.append(row_concat)
        
        grid_array = np.concatenate(grid_images, axis=0)
        grid_image = Image.fromarray(grid_array)
        
        return grid_image
'''

with open('demo.py', 'w') as f:
    f.write(demo_code)

print("✅ Demo application saved to demo.py")
print("   - Interactive controllable style transfer")
print("   - Gradio web interface")
print("   - Real-time parameter control")
print("   - Controllability grid generation")