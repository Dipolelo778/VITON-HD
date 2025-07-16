# export.py

import torch
import os

def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model + optimizer state to a .pth file.
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, path)
    print(f"✅ Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, path, device='cuda'):
    """
    Load model + optimizer state from .pth file.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"✅ Loaded checkpoint from {path} (epoch {epoch})")
    return model, optimizer, epoch, loss


def export_final_model(model, output_dir="exports", filename="viton_hd_final.pth"):
    """
    Save final trained model only (no optimizer).
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Exported final model to {save_path}")
    
