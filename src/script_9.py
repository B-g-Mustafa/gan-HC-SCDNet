# Demo application and utilities
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
        
        # Post-processing
        self.denorm = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # [-1,1] -> [0,1]
        
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
        """
        Generate style transfer with controllable parameters
        Args:
            content_image: Content image (PIL)
            style_image: Style reference image (PIL)
            style_strength: Style application intensity [0, 1]
            content_preservation: Content preservation strength [0, 1]
            num_steps: Number of diffusion steps [5, 20]
        Returns:
            Stylized image (PIL)
        """
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
        """
        Create a 3x3 grid showing different control combinations
        Demonstrates the controllability of HC-SCDNet
        """
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
                    num_steps=8  # Faster for grid generation
                )
                row_images.append(np.array(generated))
            
            # Concatenate row
            row_concat = np.concatenate(row_images, axis=1)
            grid_images.append(row_concat)
        
        # Concatenate all rows
        grid_array = np.concatenate(grid_images, axis=0)
        grid_image = Image.fromarray(grid_array)
        
        return grid_image
    
    def launch_gradio_app(self, share: bool = False):
        """Launch interactive Gradio application"""
        
        def transfer_fn(content_img, style_img, style_strength, content_preservation, steps):
            """Gradio interface function"""
            try:
                if content_img is None or style_img is None:
                    return None, "Please upload both content and style images"
                
                result = self.generate_style_transfer(
                    content_img, style_img, style_strength, content_preservation, int(steps)
                )
                return result, "Style transfer completed successfully!"
                
            except Exception as e:
                return None, f"Error: {str(e)}"
        
        def grid_fn(content_img, style_img):
            """Generate controllability grid"""
            try:
                if content_img is None or style_img is None:
                    return None, "Please upload both images"
                
                grid = self.create_control_grid(content_img, style_img)
                return grid, "Control grid generated! Rows: Style (0.2, 0.6, 1.0), Cols: Content (0.2, 0.6, 1.0)"
                
            except Exception as e:
                return None, f"Error: {str(e)}"
        
        # Create Gradio interface
        with gr.Blocks(title="HC-SCDNet: Controllable Style Transfer") as demo:
            gr.Markdown("""
            # HC-SCDNet: Hybrid Controllable Style-Content Disentanglement Network
            
            Upload a content image and style reference to generate controllable artistic style transfer.
            Adjust the sliders to control style intensity and content preservation independently.
            """)
            
            with gr.Tab("Single Generation"):
                with gr.Row():
                    with gr.Column():
                        content_input = gr.Image(type="pil", label="Content Image")
                        style_input = gr.Image(type="pil", label="Style Reference")
                        
                        style_slider = gr.Slider(0.0, 1.0, value=1.0, step=0.1, 
                                               label="Style Strength")
                        content_slider = gr.Slider(0.0, 1.0, value=1.0, step=0.1,
                                                 label="Content Preservation")
                        steps_slider = gr.Slider(5, 20, value=10, step=1,
                                               label="Generation Steps (more = higher quality)")
                        
                        generate_btn = gr.Button("Generate Style Transfer", variant="primary")
                    
                    with gr.Column():
                        output_image = gr.Image(type="pil", label="Generated Result")
                        status_text = gr.Textbox(label="Status")
                
                generate_btn.click(
                    transfer_fn,
                    inputs=[content_input, style_input, style_slider, content_slider, steps_slider],
                    outputs=[output_image, status_text]
                )
            
            with gr.Tab("Controllability Grid"):
                with gr.Row():
                    with gr.Column():
                        grid_content = gr.Image(type="pil", label="Content Image")
                        grid_style = gr.Image(type="pil", label="Style Reference")
                        grid_btn = gr.Button("Generate Control Grid", variant="secondary")
                    
                    with gr.Column():
                        grid_output = gr.Image(type="pil", label="Controllability Grid")
                        grid_status = gr.Textbox(label="Status")
                
                grid_btn.click(
                    grid_fn,
                    inputs=[grid_content, grid_style],
                    outputs=[grid_output, grid_status]
                )
            
            gr.Markdown("""
            ## Usage Tips:
            - **Style Strength**: Controls how much artistic style is applied (0.0 = no style, 1.0 = full style)
            - **Content Preservation**: Controls how much original content is preserved (0.0 = abstract, 1.0 = realistic)
            - **Steps**: More steps = higher quality but slower generation
            - **Control Grid**: Shows 9 combinations of different style/content settings
            """)
        
        demo.launch(share=share, server_name="0.0.0.0", server_port=7860)

def create_baseline_comparison():
    """
    Utility to compare HC-SCDNet with baseline methods
    """
    comparison_code = '''
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import time

class BaselineComparison:
    """Compare HC-SCDNet with baseline methods"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models (would need actual implementations)
        self.hcscdnet = None  # Load trained HC-SCDNet
        self.adain_model = None  # Load AdaIN model
        self.gatys_optimizer = None  # Gatys optimization
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def compare_methods(self, content_image: Image.Image, style_image: Image.Image):
        """Compare all methods on same input"""
        results = {}
        
        # Preprocess
        content_tensor = self.transform(content_image).unsqueeze(0).to(self.device)
        style_tensor = self.transform(style_image).unsqueeze(0).to(self.device)
        
        # HC-SCDNet (Our Method)
        if self.hcscdnet:
            start_time = time.time()
            hc_result = self.hcscdnet(content_tensor, style_tensor)
            hc_time = time.time() - start_time
            results['HC-SCDNet'] = {
                'image': self.tensor_to_pil(hc_result['generated_image']),
                'time': hc_time,
                'controllable': True
            }
        
        # AdaIN Baseline
        if self.adain_model:
            start_time = time.time() 
            adain_result = self.adain_model(content_tensor, style_tensor)
            adain_time = time.time() - start_time
            results['AdaIN'] = {
                'image': self.tensor_to_pil(adain_result),
                'time': adain_time,
                'controllable': False
            }
        
        # Gatys Optimization (would be much slower)
        # results['Gatys'] = {...}
        
        return results
    
    def tensor_to_pil(self, tensor):
        """Convert tensor to PIL image"""
        tensor = torch.clamp((tensor + 1) / 2, 0, 1)
        tensor_cpu = tensor.squeeze(0).cpu()
        image_np = tensor_cpu.permute(1, 2, 0).numpy()
        return Image.fromarray((image_np * 255).astype('uint8'))
    
    def create_comparison_figure(self, results, content_img, style_img):
        """Create comparison visualization"""
        n_methods = len(results)
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(4*(n_methods+1), 8))
        
        # Input images
        axes[0, 0].imshow(content_img)
        axes[0, 0].set_title('Content')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(style_img)
        axes[1, 0].set_title('Style')
        axes[1, 0].axis('off')
        
        # Results
        for i, (method, result) in enumerate(results.items()):
            col = i + 1
            
            # Generated image
            axes[0, col].imshow(result['image'])
            axes[0, col].set_title(f"{method}\\n{result['time']:.2f}s")
            axes[0, col].axis('off')
            
            # Info
            info_text = f"Time: {result['time']:.2f}s\\n"
            info_text += f"Controllable: {'Yes' if result['controllable'] else 'No'}"
            
            axes[1, col].text(0.1, 0.5, info_text, fontsize=10, 
                            transform=axes[1, col].transAxes)
            axes[1, col].axis('off')
        
        plt.tight_layout()
        return fig
    '''
    
    return comparison_code

# Create demo script
demo_script = '''
from demo import HCSCDNetDemo
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--share', action='store_true', help='Share gradio app publicly')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = HCSCDNetDemo(args.model_path, args.device)
    
    # Launch interactive app
    print("Launching HC-SCDNet interactive demo...")
    demo.launch_gradio_app(share=args.share)

if __name__ == "__main__":
    main()
'''

with open('demo.py', 'w') as f:
    f.write(demo_code)

with open('run_demo.py', 'w') as f:
    f.write(demo_script)

baseline_comparison = create_baseline_comparison()
with open('baseline_comparison.py', 'w') as f:
    f.write(baseline_comparison)

print("✅ Demo and Utilities saved:")
print("   - demo.py: Interactive Gradio application")
print("   - run_demo.py: Demo launcher script")
print("   - baseline_comparison.py: Method comparison utilities")
print("   - Controllability grid generation")
print("   - Real-time style transfer interface")