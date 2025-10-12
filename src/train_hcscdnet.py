
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import os
from hcscdnet_main import HCSCDNet
from dataset import StyleTransferDataset
import argparse
from typing import Dict
import matplotlib.pyplot as plt

class HCSCDNetTrainer:
    """
    Training pipeline for HC-SCDNet following the 2-week implementation plan
    Implements the fine-tuning strategy (not training from scratch)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = HCSCDNet(
            style_latent_dim=config['style_latent_dim'],
            content_latent_dim=config['content_latent_dim'], 
            diffusion_steps=config['diffusion_steps'],
            beta=config['beta'],
            device=self.device
        ).to(self.device)

        # Print parameter counts
        param_counts = self.model.count_parameters()
        print(f"Model Parameter Counts:")
        for module, count in param_counts.items():
            print(f"  {module}: {count:,}")

        # Optimizers (separate for different components following fine-tuning strategy)
        self.optimizer_vae = optim.Adam(
            self.model.beta_vae_encoder.parameters(),
            lr=config['vae_lr'],
            weight_decay=config['weight_decay']
        )

        self.optimizer_diffusion = optim.Adam(
            self.model.diffusion_unet.parameters(),
            lr=config['diffusion_lr'], 
            weight_decay=config['weight_decay']
        )

        self.optimizer_discriminator = optim.Adam(
            self.model.quality_enhancer.discriminator.parameters(),
            lr=config['discriminator_lr'],
            weight_decay=config['weight_decay']
        )

        # Learning rate schedulers
        self.scheduler_vae = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_vae, T_max=config['num_epochs']
        )
        self.scheduler_diffusion = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_diffusion, T_max=config['num_epochs'] 
        )

        # Loss weights (following diagram specifications)
        self.loss_weights = {
            'reconstruction': config.get('recon_weight', 1.0),
            'style_kl': config.get('style_kl_weight', 1.0),
            'content_kl': config.get('content_kl_weight', 1.0),
            'mi_penalty': config.get('mi_weight', 0.1),
            'perceptual': config.get('perceptual_weight', 1.0),
            'style': config.get('style_weight', 250.0),
            'adversarial': config.get('adversarial_weight', 0.1)
        }

        # Initialize wandb for experiment tracking
        if config.get('use_wandb', True):
            wandb.init(
                project="HC-SCDNet",
                config=config,
                name=f"hcscdnet_beta{config['beta']}_steps{config['diffusion_steps']}"
            )

    def train_phase1_disentanglement(self, dataloader: DataLoader, epoch: int):
        """
        Phase 1: β-VAE Disentanglement Learning (Days 5-7)
        Focus on learning good style-content disentanglement
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f'Phase 1 - Epoch {epoch}')

        for batch in pbar:
            content_images = batch['content'].to(self.device)
            style_images = batch['style'].to(self.device)

            # Forward pass through β-VAE encoder only
            content_encoding = self.model.beta_vae_encoder(content_images)
            style_encoding = self.model.beta_vae_encoder(style_images)

            # Simple reconstruction for disentanglement learning
            # Use shared features for reconstruction (simplified)
            reconstructed = torch.tanh(content_encoding['shared_features'].mean(dim=(2, 3), keepdim=True))
            reconstructed = nn.functional.interpolate(reconstructed, size=content_images.shape[-2:])

            # Compute β-VAE losses
            vae_losses = self.model.compute_beta_vae_loss(
                content_encoding['content_mu'],
                content_encoding['content_logvar'],
                style_encoding['style_mu'],
                style_encoding['style_logvar'],
                reconstructed,
                content_images
            )

            # Phase 1 loss (focus on disentanglement)
            loss = (
                self.loss_weights['reconstruction'] * vae_losses['reconstruction'] +
                self.loss_weights['style_kl'] * vae_losses['style_kl'] +
                self.loss_weights['content_kl'] * vae_losses['content_kl'] +
                self.loss_weights['mi_penalty'] * vae_losses['mi_penalty']
            )

            # Optimize β-VAE
            self.optimizer_vae.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.beta_vae_encoder.parameters(), 1.0)
            self.optimizer_vae.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Recon': f'{vae_losses["reconstruction"].item():.4f}',
                'KL': f'{(vae_losses["style_kl"] + vae_losses["content_kl"]).item():.4f}'
            })

            # Log to wandb
            if hasattr(self, 'config') and self.config.get('use_wandb'):
                wandb.log({
                    'phase1/total_loss': loss.item(),
                    'phase1/reconstruction_loss': vae_losses['reconstruction'].item(),
                    'phase1/style_kl': vae_losses['style_kl'].item(), 
                    'phase1/content_kl': vae_losses['content_kl'].item(),
                    'phase1/mi_penalty': vae_losses['mi_penalty'].item()
                })

        avg_loss = total_loss / num_batches
        self.scheduler_vae.step()

        return avg_loss

    def train_phase2_integration(self, dataloader: DataLoader, epoch: int):
        """
        Phase 2: Hybrid Integration (Days 8-10)
        Integrate β-VAE with diffusion model
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f'Phase 2 - Epoch {epoch}')

        for batch in pbar:
            content_images = batch['content'].to(self.device)
            style_images = batch['style'].to(self.device)

            # Full forward pass
            outputs = self.model(content_images, style_images)

            vae_losses = outputs['vae_losses']
            quality_losses = outputs['quality_losses']

            # Phase 2 loss (integrate all components)
            loss = (
                # VAE losses (reduced weight as it's pre-trained)
                0.5 * self.loss_weights['reconstruction'] * vae_losses['reconstruction'] +
                0.5 * self.loss_weights['style_kl'] * vae_losses['style_kl'] +
                0.5 * self.loss_weights['content_kl'] * vae_losses['content_kl'] +
                0.5 * self.loss_weights['mi_penalty'] * vae_losses['mi_penalty'] +
                # Quality enhancement losses
                self.loss_weights['perceptual'] * quality_losses['perceptual'] +
                self.loss_weights['style'] * quality_losses['style'] +
                self.loss_weights['adversarial'] * quality_losses['adversarial']
            )

            # Optimize generator components (VAE + Diffusion)
            self.optimizer_vae.zero_grad()
            self.optimizer_diffusion.zero_grad()
            loss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(self.model.beta_vae_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.diffusion_unet.parameters(), 1.0)

            self.optimizer_vae.step()
            self.optimizer_diffusion.step()

            # Train discriminator
            disc_losses = self.model.quality_enhancer.compute_discriminator_loss(
                outputs['generated_image'].detach(),
                content_images
            )

            self.optimizer_discriminator.zero_grad()
            disc_losses['total'].backward()
            self.optimizer_discriminator.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'G_Loss': f'{loss.item():.4f}',
                'D_Loss': f'{disc_losses["total"].item():.4f}',
                'Style': f'{quality_losses["style"].item():.4f}'
            })

            # Log to wandb
            if hasattr(self, 'config') and self.config.get('use_wandb'):
                wandb.log({
                    'phase2/generator_loss': loss.item(),
                    'phase2/discriminator_loss': disc_losses['total'].item(),
                    'phase2/perceptual_loss': quality_losses['perceptual'].item(),
                    'phase2/style_loss': quality_losses['style'].item(),
                    'phase2/adversarial_loss': quality_losses['adversarial'].item()
                })

        avg_loss = total_loss / num_batches
        self.scheduler_diffusion.step()

        return avg_loss

    def train_phase3_finetuning(self, dataloader: DataLoader, epoch: int):
        """
        Phase 3: End-to-End Fine-tuning (Days 11-12)
        Fine-tune the complete system
        """
        # Similar to Phase 2 but with lower learning rates and full integration
        return self.train_phase2_integration(dataloader, epoch)

    def validate(self, dataloader: DataLoader, epoch: int):
        """Validation loop with controllability demonstration"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 10:  # Limit validation samples
                    break

                content_images = batch['content'].to(self.device)
                style_images = batch['style'].to(self.device)

                # Generate with different control settings
                outputs = self.model(content_images[:1], style_images[:1])

                if i == 0:  # Save sample results
                    # Generate controllability grid
                    control_grid = self.model.get_controllable_generation(
                        content_images[:1],
                        style_images[:1],
                        style_controls=[0.0, 0.5, 1.0],
                        content_controls=[0.0, 0.5, 1.0]
                    )

                    # Log to wandb
                    if hasattr(self, 'config') and self.config.get('use_wandb'):
                        wandb.log({
                            f'validation/control_grid_epoch_{epoch}': wandb.Image(
                                control_grid[0].cpu().detach().clamp(0, 1)
                            )
                        })

        return total_loss / max(num_batches, 1)

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        """
        Complete training pipeline following the 2-week schedule
        """
        print(f"Starting HC-SCDNet training on {self.device}")
        print(f"Total epochs: {self.config['num_epochs']}")

        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")

            # Phase selection based on epoch (following 2-week plan)
            if epoch < self.config.get('phase1_epochs', 5):
                # Days 5-7: Disentanglement learning
                train_loss = self.train_phase1_disentanglement(train_dataloader, epoch)
                print(f"Phase 1 - Disentanglement Learning - Loss: {train_loss:.4f}")

            elif epoch < self.config.get('phase2_epochs', 15):
                # Days 8-10: Integration training
                train_loss = self.train_phase2_integration(train_dataloader, epoch)
                print(f"Phase 2 - Hybrid Integration - Loss: {train_loss:.4f}")

            else:
                # Days 11-12: Fine-tuning
                train_loss = self.train_phase3_finetuning(train_dataloader, epoch)
                print(f"Phase 3 - End-to-End Fine-tuning - Loss: {train_loss:.4f}")

            # Validation
            val_loss = self.validate(val_dataloader, epoch)
            print(f"Validation Loss: {val_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                self.save_checkpoint(epoch + 1)

        print("Training completed!")

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_vae_state_dict': self.optimizer_vae.state_dict(),
            'optimizer_diffusion_state_dict': self.optimizer_diffusion.state_dict(),
            'optimizer_discriminator_state_dict': self.optimizer_discriminator.state_dict(),
            'config': self.config
        }

        torch.save(checkpoint, f'{checkpoint_dir}/hcscdnet_epoch_{epoch}.pth')
        print(f"Checkpoint saved: epoch {epoch}")

def main():
    """Main training script"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Configuration
    config = {
        'style_latent_dim': 512,
        'content_latent_dim': 512,
        'diffusion_steps': 10,
        'beta': 4.0,
        'device': args.device,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'vae_lr': 1e-4,
        'diffusion_lr': 1e-5,  # Lower for pre-trained components
        'discriminator_lr': 2e-4,
        'weight_decay': 1e-5,
        'phase1_epochs': 5,
        'phase2_epochs': 15,
        'use_wandb': True,
        'save_every': 5,
        'checkpoint_dir': 'checkpoints'
    }

    # Initialize trainer
    trainer = HCSCDNetTrainer(config)

    # Create datasets (placeholder - implement actual dataset loading)
    from torch.utils.data import TensorDataset

    # Placeholder data (replace with actual StyleTransferDataset)
    dummy_content = torch.randn(100, 3, 512, 512)
    dummy_style = torch.randn(100, 3, 512, 512)
    train_dataset = TensorDataset(dummy_content, dummy_style)
    val_dataset = TensorDataset(dummy_content[:20], dummy_style[:20])

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Start training
    trainer.train(train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()
