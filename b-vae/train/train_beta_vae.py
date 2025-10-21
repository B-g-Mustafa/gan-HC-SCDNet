# train_beta_vae.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn
import torchvision.utils as vutils
from data import SimpleImageFolder
from beta_vae import BetaVAE
from PIL import Image
import numpy as np
import itertools

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--style_dir', type=str, required=True, help='Path to WikiArt style images')
    p.add_argument('--content_dir', type=str, required=True, help='Path to COCO content images')
    p.add_argument('--val_style_dir', type=str, default=None, help='Validation style images')
    p.add_argument('--val_content_dir', type=str, default=None, help='Validation content images')
    p.add_argument('--image_size', type=int, default=256)
    p.add_argument('--style_dim', type=int, default=128)
    p.add_argument('--content_dim', type=int, default=128)
    p.add_argument('--batch_style', type=int, default=16)
    p.add_argument('--batch_content', type=int, default=16)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--save_every', type=int, default=1)
    return p.parse_args()

# Loss helpers
def kl_loss(mu, lv):
    # lv is logvar
    return -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())

#Dynamic beta schedule
def get_beta(epoch):
    if epoch < 3:       # warm-up
        return 1.0
    elif epoch < 7:     # linear increase
        return 1.0 + (epoch - 2) * 0.75   # 1 → 4 over epochs 3–6
    else:
        return 4.0


def reconstruction_loss(x, recon):
    return torch.mean(torch.abs(x - recon))  # L1

def save_image_tensor(tensor, path, nrow=4):
    vutils.save_image(tensor, path, nrow=nrow, normalize=True, value_range=(0,1))

def latent_traversal_and_save(model, samples, out_dir, device, n_steps=8):
    # samples: tensor [N,3,H,W] (we will take first two images)
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        imgs = samples.to(device)
        mu_s, lv_s, mu_c, lv_c = model._forward_stats(imgs)
        # pick first two images for visualization
        a_idx = 0; b_idx = 1 if imgs.shape[0] > 1 else 0
        s_a = mu_s[a_idx:a_idx+1]; c_b = mu_c[b_idx:b_idx+1]
        # traverse style from s_a -> s_other or along principal axes
        traversals = []
        # linear interpolate style latent between s_a and s_b
        if imgs.shape[0] > 1:
            s_b = mu_s[b_idx:b_idx+1]
            for alpha in np.linspace(0,1,n_steps):
                s_mix = s_a * (1-alpha) + s_b * alpha
                dec = model.decode_from_means(s_mix, c_b)
                traversals.append(dec.cpu())
        else:
            # vary along +/- principal directions
            for dim in range(min(4, model.style_dim)):
                for scale in np.linspace(-2,2,n_steps):
                    s_var = s_a.clone()
                    s_var[0,dim] += float(scale)
                    dec = model.decode_from_means(s_var, c_b)
                    traversals.append(dec.cpu())
        grid = torch.cat(traversals, dim=0)
        save_image_tensor(grid, os.path.join(out_dir, 'traversal_grid.png'), nrow=n_steps)

def find_latest_checkpoint(save_dir):
    """Find the latest checkpoint file in the save directory."""
    if not os.path.exists(save_dir):
        return None
    
    # Look for checkpoint files (excluding best model)
    checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith('beta_vae_epoch') and f.endswith('.pth')]
    
    if not checkpoint_files:
        return None
    
    # Sort by epoch number and get the latest
    checkpoint_files.sort(key=lambda x: int(x.split('epoch')[1].replace('.pth', '')))
    latest_checkpoint = checkpoint_files[-1]
    
    return os.path.join(save_dir, latest_checkpoint)

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model and optimizer state from a checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        epoch = checkpoint['epoch']
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
        print(f"  Resuming from epoch {epoch + 1}")
        return epoch
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return 0

def train():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    style_ds = SimpleImageFolder(args.style_dir, image_size=args.image_size)
    content_ds = SimpleImageFolder(args.content_dir, image_size=args.image_size)
    style_loader = DataLoader(style_ds, batch_size=args.batch_style, shuffle=True, num_workers=2, drop_last=True)
    content_loader = DataLoader(content_ds, batch_size=args.batch_content, shuffle=True, num_workers=2, drop_last=True)
    
    # Load validation datasets if provided
    val_loader = None
    best_loss = float('inf')
    if args.val_style_dir and args.val_content_dir:
        try:
            val_style_ds = SimpleImageFolder(args.val_style_dir, image_size=args.image_size)
            val_content_ds = SimpleImageFolder(args.val_content_dir, image_size=args.image_size)
            # Create a combined validation loader (we'll iterate through content and cycle style)
            val_style_loader = DataLoader(val_style_ds, batch_size=args.batch_style, shuffle=False, num_workers=2, drop_last=True)
            val_content_loader = DataLoader(val_content_ds, batch_size=args.batch_content, shuffle=False, num_workers=2, drop_last=True)
            val_loader = (val_style_loader, val_content_loader)
            print(f"Loaded validation datasets. Style: {len(val_style_ds)}, Content: {len(val_content_ds)}")
        except Exception as e:
            print(f"Warning: Could not load validation datasets: {e}")
            val_loader = None

    # Combined iterator (cycle shorter dataset)
    style_iter = itertools.cycle(style_loader)
    content_iter = iter(content_loader)
    steps_per_epoch = len(content_loader)

    model = BetaVAE(img_size=args.image_size, style_dim=args.style_dim, content_dim=args.content_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Try to load the latest checkpoint
    start_epoch = 1
    latest_checkpoint = find_latest_checkpoint(args.save_dir)
    if latest_checkpoint:
        start_epoch = load_checkpoint(model, optimizer, latest_checkpoint, device) + 1
    else:
        print("No checkpoint found. Starting training from epoch 1.")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        beta = get_beta(epoch)
        running_loss = 0

        for step in range(steps_per_epoch):
            style_imgs = next(style_iter).to(device)
            try:
                content_imgs = next(content_iter)
            except StopIteration:
                content_iter = iter(content_loader)
                content_imgs = next(content_iter)
            content_imgs = content_imgs.to(device)

            mu_s, lv_s, _, _ = model._forward_stats(style_imgs)
            _, _, mu_c, lv_c = model._forward_stats(content_imgs)

            z_s = model.reparam(mu_s, lv_s)
            z_c = model.reparam(mu_c, lv_c)
            recon = model.decode_from_latent(z_s, z_c)

            L_rec = torch.mean(torch.abs(recon - content_imgs))
            KL_s = -0.5 * torch.mean(1 + lv_s - mu_s.pow(2) - lv_s.exp())
            KL_c = -0.5 * torch.mean(1 + lv_c - mu_c.pow(2) - lv_c.exp())
            loss = L_rec + beta * (KL_s + KL_c)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 100 == 0:
                print(f"Epoch {epoch}/{args.epochs} | Step {step}/{steps_per_epoch} | "
                      f"Loss {loss.item():.4f} (rec {L_rec.item():.4f}, kl {(KL_s+KL_c).item():.4f}, beta={beta:.2f})")

        print(f"Epoch {epoch} complete. Avg loss: {running_loss / steps_per_epoch:.4f}")
        
        # checkpoint
        if epoch % args.save_every == 0:
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'args': vars(args)
            }
            torch.save(ckpt, os.path.join(args.save_dir, f'beta_vae_epoch{epoch}.pth'))
            print(f"Saved checkpoint epoch {epoch}")

        # (optional) evaluate on val set
        if val_loader:
            try:
                model.eval()
                val_loss = 0.0
                val_steps = 0
                val_style_loader, val_content_loader = val_loader
                val_style_iter = itertools.cycle(val_style_loader)
                
                with torch.no_grad():
                    for val_content_batch in val_content_loader:
                        val_style_batch = next(val_style_iter)
                        
                        val_content_batch = val_content_batch.to(device)
                        val_style_batch = val_style_batch.to(device)
                        
                        # Get latent statistics
                        mu_s, lv_s, _, _ = model._forward_stats(val_style_batch)
                        _, _, mu_c, lv_c = model._forward_stats(val_content_batch)
                        
                        # Reconstruct
                        z_s = model.reparam(mu_s, lv_s)
                        z_c = model.reparam(mu_c, lv_c)
                        recon = model.decode_from_latent(z_s, z_c)
                        
                        # Calculate losses
                        rec_loss = torch.mean(torch.abs(recon - val_content_batch))
                        KL_s = -0.5 * torch.mean(1 + lv_s - mu_s.pow(2) - lv_s.exp())
                        KL_c = -0.5 * torch.mean(1 + lv_c - mu_c.pow(2) - lv_c.exp())
                        val_step_loss = rec_loss + beta * (KL_s + KL_c)
                        
                        val_loss += val_step_loss.item()
                        val_steps += 1
                
                avg_val_loss = val_loss / max(val_steps, 1)
                print(f"Validation loss: {avg_val_loss:.4f}")
                
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    torch.save(ckpt, os.path.join(args.save_dir, 'beta_vae_best.pth'))
                    print(f"✓ Saved best model with val loss: {best_loss:.4f}")
                
                model.train()  # Switch back to training mode
            except Exception as e:
                print(f"Warning: Validation failed with error: {e}")
                import traceback
                traceback.print_exc()

if __name__ == '__main__':
    train()
