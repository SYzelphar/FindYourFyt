"""Image/text encoders behind one interface (PyTorch).

    fashion_clip  FashionCLIP (CLIP ViT-B/32 fine-tuned on ~800k fashion products), 512-d image + text
    clip          OpenAI CLIP ViT-B/32, 512-d image + text            (general-purpose baseline)
    vgg16         VGG16 ImageNet conv features, GAP over block5, 512-d (the v1 model, baseline)

All outputs are L2-normalised float32. Product photos are 2:3 portraits, so images are
padded to a square with their own background colour instead of centre-cropped (a crop
would cut off collars and hems).
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config as C

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def pad_to_square(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if w == h:
        return img
    corners = [img.getpixel(p) for p in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))]
    bg = tuple(int(np.median([c[i] for c in corners])) for i in range(3))
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), bg)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


class ImageFiles(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = list(paths), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.transform(pad_to_square(im))


def _normalize(x: torch.Tensor) -> np.ndarray:
    return torch.nn.functional.normalize(x.float(), dim=-1).cpu().numpy().astype(np.float32)


class Encoder:
    name: str
    dim: int
    supports_text = False

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def image_transform(self):
        raise NotImplementedError

    def _image_features(self, pixels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def encode_images(self, paths, batch_size=128, num_workers=8, progress=True) -> np.ndarray:
        loader = DataLoader(ImageFiles(paths, self.image_transform()), batch_size=batch_size,
                            num_workers=num_workers, pin_memory=self.device == "cuda",
                            persistent_workers=num_workers > 0)
        out, done = [], 0
        for batch in loader:
            out.append(_normalize(self._image_features(batch.to(self.device, non_blocking=True))))
            done += len(batch)
            if progress and (len(out) % 20 == 0 or done == len(loader.dataset)):
                print(f"    {self.name}: {done}/{len(loader.dataset)} images")
        return np.concatenate(out)

    def encode_text(self, texts, batch_size=256) -> np.ndarray:
        raise NotImplementedError(f"{self.name} has no text encoder")


class ClipEncoder(Encoder):
    """CLIP-family models from Hugging Face transformers."""
    supports_text = True
    dim = 512

    def __init__(self, name: str, hf_id: str, device: str | None = None):
        super().__init__(device)
        from transformers import CLIPModel, CLIPProcessor
        self.name = name
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(hf_id, dtype=dtype).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(hf_id)
        self.dtype = dtype

    def image_transform(self):
        ip = self.processor.image_processor
        return transforms.Compose([
            transforms.Resize((C.ENCODER_IMAGE_SIZE, C.ENCODER_IMAGE_SIZE),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(ip.image_mean, ip.image_std),
        ])

    def _image_features(self, pixels):
        return self.model.get_image_features(pixel_values=pixels.to(self.dtype))

    @torch.inference_mode()
    def encode_text(self, texts, batch_size=256) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            tok = self.processor.tokenizer(list(texts[i:i + batch_size]), padding=True, truncation=True,
                                           max_length=77, return_tensors="pt").to(self.device)
            out.append(_normalize(self.model.get_text_features(**tok)))
        return np.concatenate(out)


class Vgg16Encoder(Encoder):
    name = "vgg16"
    dim = 512

    def __init__(self, device: str | None = None):
        super().__init__(device)
        from torchvision.models import VGG16_Weights, vgg16
        self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(self.device).eval()

    def image_transform(self):
        return transforms.Compose([
            transforms.Resize((C.ENCODER_IMAGE_SIZE, C.ENCODER_IMAGE_SIZE),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def _image_features(self, pixels):
        return self.model(pixels).mean(dim=(2, 3))   # block5_pool -> global average pooling


def load_encoder(name: str, device: str | None = None) -> Encoder:
    if name == "vgg16":
        return Vgg16Encoder(device)
    return ClipEncoder(name, C.ENCODERS[name], device)


def item_text(row: dict) -> str:
    """Text used for the text tower and for steering-free catalog search."""
    return (f"{row['prod_name']}. {row['colour_group_name']} {row['product_type_name']}, "
            f"{row['graphical_appearance_name']}. {row['detail_desc']}").strip()
