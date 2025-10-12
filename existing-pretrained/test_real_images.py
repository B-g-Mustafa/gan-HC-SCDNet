
"""
Test HC-SCDNet with your own images using pre-trained models
Run this BEFORE fine-tuning to verify everything works
"""

from test_pretrained_models import PretrainedHCSCDNet
from PIL import Image
import torchvision.transforms as transforms
import torch
import os

def test_with_real_images(content_path: str, style_path: str, output_dir: str = 'test_results'):
    """
    Test style transfer with real images

    Args:
        content_path: Path to content image
        style_path: Path to style reference image
        output_dir: Directory to save results
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup
    device = 'mps' 
    #if torch.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Loading pre-trained models...")

    model = PretrainedHCSCDNet(device=device)
    model.eval()

    # Load images
    print(f"\nLoading images:")
    print(f"  Content: {content_path}")
    print(f"  Style: {style_path}")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    content_img = Image.open(content_path).convert('RGB')
    style_img = Image.open(style_path).convert('RGB')

    content_tensor = transform(content_img).unsqueeze(0).to(device)
    style_tensor = transform(style_img).unsqueeze(0).to(device)

    # Save input images for reference
    content_img.save(f'{output_dir}/input_content.jpg')
    style_img.save(f'{output_dir}/input_style.jpg')

    # Test different control settings
    print(f"\nGenerating style transfers with different control settings...")
    print(f"Results will be saved to: {output_dir}/")
    print()

    test_configs = [
        # (name, style_strength, content_preservation, steps, description)
        # ("light_style", 0.3, 1.0, 15, "Light style application, full content preservation"),
        # ("medium_style", 0.6, 1.0, 15, "Medium style, full content preservation"),
        # ("full_style", 1.0, 1.0, 15, "Full style, full content preservation"),
        # ("artistic", 1.0, 0.6, 20, "Strong style, moderate content - most artistic"),
        # ("balanced", 0.7, 0.7, 15, "Balanced style and content"),
        ("europe-own-config-20", 0.8, 0.6, 15, "Balanced style and content"),
        ("abstract", 1.0, 0.3, 20, "Maximum style, minimal content - abstract"),
    ]

    import time

    for name, style_str, content_pres, steps, desc in test_configs:
        print(f"Generating '{name}':")
        print(f"  {desc}")
        print(f"  Style strength: {style_str}, Content: {content_pres}, Steps: {steps}")

        start_time = time.time()

        with torch.no_grad():
            result = model.controllable_style_transfer(
                content_tensor,
                style_tensor,
                style_strength=style_str,
                content_preservation=content_pres,
                num_inference_steps=steps,
                style_prompt=f"artistic painting, {desc}"
            )

        elapsed = time.time() - start_time

        # Convert to PIL and save
        result_img = (result[0] + 1) / 2  # [-1,1] -> [0,1]
        result_img = result_img.clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        result_pil = Image.fromarray((result_img * 255).astype('uint8'))
        result_pil.save(f'{output_dir}/result_{name}.jpg')

        print(f"  ✅ Saved: {output_dir}/result_{name}.jpg ({elapsed:.2f}s)")
        print()

    print("="*60)
    print("Testing Complete!")
    print(f"Check {output_dir}/ for all results")
    print()
    print("Files created:")
    print(f"  • input_content.jpg - Your content image")
    print(f"  • input_style.jpg - Your style reference")
    for name, _, _, _, _ in test_configs:
        print(f"  • result_{name}.jpg")
    print()
    print("Next steps:")
    print("1. Review the results")
    print("2. If quality is acceptable, this validates the architecture")
    print("3. Fine-tuning will improve quality significantly")
    print("4. You can use these as baseline for comparison")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test HC-SCDNet with real images")
    parser.add_argument('--content', type=str, required=True, help='Path to content image')
    parser.add_argument('--style', type=str, required=True, help='Path to style image')
    parser.add_argument('--output', type=str, default='test_results', help='Output directory')

    args = parser.parse_args()

    test_with_real_images(args.content, args.style, args.output)
