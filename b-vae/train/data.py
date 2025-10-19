# data.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class SimpleImageFolder(Dataset):
    """
    Simple folder dataset. Expects JPG/PNG images in `root`.
    """
    def __init__(self, root, image_size=256):
        super().__init__()
        self.root = root
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        self.files = [os.path.join(root,f) for f in os.listdir(root) if f.lower().endswith(exts)]
        assert len(self.files) > 0, f"No images found in {root}"
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.CenterCrop(image_size),
            T.ToTensor(),                 # [0,1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)
