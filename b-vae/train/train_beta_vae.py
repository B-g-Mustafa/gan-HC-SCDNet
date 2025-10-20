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

def train():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load datasets
    style_ds = SimpleImageFolder(args.style_dir, image_size=args.image_size)
    content_ds = SimpleImageFolder(args.content_dir, image_size=args.image_size)
    style_loader = DataLoader(style_ds, batch_size=args.batch_style, shuffle=True, num_workers=6, drop_last=True)
    content_loader = DataLoader(content_ds, batch_size=args.batch_content, shuffle=True, num_workers=6, drop_last=True)

    # Combined iterator (cycle shorter dataset)
    style_iter = itertools.cycle(style_loader)
    content_iter = iter(content_loader)
    steps_per_epoch = len(content_loader)

    model = BetaVAE(img_size=args.image_size, style_dim=args.style_dim, content_dim=args.content_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
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
            val_loss = 0.0
            for vb in val_loader:
                vb = vb.to(device)
                recon, mu_s, lv_s, mu_c, lv_c = model(vb)
                rec_loss = reconstruction_loss(vb, recon)
                kls = kl_loss(mu_s, lv_s) + kl_loss(mu_c, lv_c)
                val_loss += (rec_loss + args.beta * kls).item()
            val_loss /= len(val_loader)
            print(f"Validation loss: {val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(ckpt, os.path.join(args.save_dir, 'beta_vae_best.pth'))
                print("Saved best model.")

if __name__ == '__main__':
    train()
