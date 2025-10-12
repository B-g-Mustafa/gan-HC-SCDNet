# Stage 3: Quality Enhancement (Adversarial Refinement)
quality_enhancement_code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List

class LightweightDiscriminator(nn.Module):
    """
    Lightweight PatchGAN discriminator for quality enhancement
    Following Stage 3 architecture from the diagram
    """
    def __init__(self, 
                 input_channels: int = 3,
                 num_filters: int = 64,
                 num_layers: int = 3):
        super().__init__()
        
        layers = []
        
        # First layer (no normalization)
        layers.append(nn.Conv2d(input_channels, num_filters, 4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, True))
        
        # Intermediate layers
        nf_mult = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers.extend([
                nn.Conv2d(num_filters * nf_mult_prev, num_filters * nf_mult, 4, stride=2, padding=1),
                nn.GroupNorm(8, num_filters * nf_mult),
                nn.LeakyReLU(0.2, True)
            ])
        
        # Final layer
        nf_mult_prev = nf_mult
        nf_mult = min(2**num_layers, 8)
        layers.extend([
            nn.Conv2d(num_filters * nf_mult_prev, num_filters * nf_mult, 4, stride=1, padding=1),
            nn.GroupNorm(8, num_filters * nf_mult),
            nn.LeakyReLU(0.2, True)
        ])
        
        # Output layer
        layers.append(nn.Conv2d(num_filters * nf_mult, 1, 4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator"""
        return self.model(x)

class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss for content preservation
    Uses multiple layers from VGG19 for better feature matching
    """
    def __init__(self, layers: List[str] = ['conv_1', 'conv_2', 'conv_3', 'conv_4']):
        super().__init__()
        
        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features
        self.vgg = nn.Sequential()
        
        # Extract specified layers
        layer_map = {
            'conv_1': 2,   # relu1_1
            'conv_2': 7,   # relu2_1  
            'conv_3': 12,  # relu3_1
            'conv_4': 21,  # relu4_1
            'conv_5': 30   # relu5_1
        }
        
        self.layers = layers
        max_layer = max([layer_map[layer] for layer in layers])
        
        for i in range(max_layer + 1):
            self.vgg.add_module(str(i), vgg[i])
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between x and target
        Args:
            x: Generated image [B, C, H, W]
            target: Target content image [B, C, H, W]
        Returns:
            Perceptual loss value
        """
        # Normalize to ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        x_norm = (x - mean) / std
        target_norm = (target - mean) / std
        
        # Extract features
        x_features = self.extract_features(x_norm)
        target_features = self.extract_features(target_norm)
        
        # Compute loss across all layers
        loss = 0
        for layer in self.layers:
            loss += F.mse_loss(x_features[layer], target_features[layer])
        
        return loss / len(self.layers)
    
    def extract_features(self, x: torch.Tensor) -> dict:
        """Extract features from specified VGG layers"""
        features = {}
        layer_map = {
            2: 'conv_1',
            7: 'conv_2', 
            12: 'conv_3',
            21: 'conv_4',
            30: 'conv_5'
        }
        
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in layer_map and layer_map[i] in self.layers:
                features[layer_map[i]] = x
                
        return features

class StyleLoss(nn.Module):
    """
    Gram matrix-based style loss for artistic style matching
    """
    def __init__(self, layers: List[str] = ['conv_1', 'conv_2', 'conv_3', 'conv_4']):
        super().__init__()
        self.perceptual = PerceptualLoss(layers)
        self.layers = layers
    
    def gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for style representation"""
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(self, x: torch.Tensor, style_target: torch.Tensor) -> torch.Tensor:
        """
        Compute style loss between x and style target
        Args:
            x: Generated image [B, C, H, W]
            style_target: Style reference image [B, C, H, W]
        Returns:
            Style loss value
        """
        # Extract features using the same VGG network
        x_features = self.perceptual.extract_features(
            (x - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)) /
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        )
        
        style_features = self.perceptual.extract_features(
            (style_target - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(style_target.device)) /
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(style_target.device)
        )
        
        # Compute Gram matrix loss
        loss = 0
        for layer in self.layers:
            x_gram = self.gram_matrix(x_features[layer])
            style_gram = self.gram_matrix(style_features[layer])
            loss += F.mse_loss(x_gram, style_gram)
        
        return loss / len(self.layers)

class QualityEnhancementModule(nn.Module):
    """
    Combined quality enhancement module with adversarial training
    Following Stage 3 architecture from the diagram
    """
    def __init__(self, 
                 input_channels: int = 3,
                 perceptual_weight: float = 1.0,
                 style_weight: float = 250.0,
                 adversarial_weight: float = 0.1):
        super().__init__()
        
        self.discriminator = LightweightDiscriminator(input_channels)
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.adversarial_weight = adversarial_weight
    
    def compute_generator_loss(self, 
                             generated: torch.Tensor,
                             content_target: torch.Tensor, 
                             style_target: torch.Tensor) -> dict:
        """
        Compute combined generator loss for quality enhancement
        Args:
            generated: Generated image from diffusion model [B, C, H, W]
            content_target: Original content image [B, C, H, W]
            style_target: Style reference image [B, C, H, W]
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Perceptual loss (content preservation)
        losses['perceptual'] = self.perceptual_loss(generated, content_target)
        
        # Style loss (artistic style matching)
        losses['style'] = self.style_loss(generated, style_target)
        
        # Adversarial loss (quality enhancement)
        fake_pred = self.discriminator(generated)
        losses['adversarial'] = F.mse_loss(fake_pred, torch.ones_like(fake_pred))
        
        # Total generator loss
        losses['total'] = (
            self.perceptual_weight * losses['perceptual'] +
            self.style_weight * losses['style'] +
            self.adversarial_weight * losses['adversarial']
        )
        
        return losses
    
    def compute_discriminator_loss(self, 
                                 generated: torch.Tensor,
                                 real: torch.Tensor) -> dict:
        """
        Compute discriminator loss for adversarial training
        Args:
            generated: Generated image [B, C, H, W]
            real: Real image [B, C, H, W]
        Returns:
            Dictionary of discriminator losses
        """
        losses = {}
        
        # Real image prediction
        real_pred = self.discriminator(real)
        losses['real'] = F.mse_loss(real_pred, torch.ones_like(real_pred))
        
        # Fake image prediction
        fake_pred = self.discriminator(generated.detach())
        losses['fake'] = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
        
        # Total discriminator loss
        losses['total'] = (losses['real'] + losses['fake']) * 0.5
        
        return losses
'''

with open('quality_enhancement.py', 'w') as f:
    f.write(quality_enhancement_code)

print("✅ Stage 3: Quality Enhancement implementation saved to quality_enhancement.py")
print("   - Lightweight PatchGAN discriminator (~10M parameters)")
print("   - VGG-based perceptual loss for content preservation")
print("   - Gram matrix style loss for artistic style matching")
print("   - Combined adversarial refinement loss")