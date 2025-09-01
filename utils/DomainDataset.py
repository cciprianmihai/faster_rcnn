import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data.dataset import Dataset


class DomainDataset(Dataset):
    """
    Loads image and binary mask pairs, applies identical resizing + random rotations,
    then finds contours in the mask to produce bounding boxes. Preserves original
    channel count of each image.
    """

    def __init__(
        self,
        root_dir,
        do_rotate=True,
        use_sam=False,
        domain_label=0,
        img_size=256,
        img_extension=".png",
        mask_extension=".png",
        class_id=1,
        transforms_fn=None,
        length=None
    ):
        super().__init__()
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "masks")
        self.image_files = [
            f for f in os.listdir(self.images_dir) if f.endswith(img_extension)
        ]
        random.shuffle(self.image_files)
        self.do_rotate = do_rotate
        self.use_sam = use_sam
        self.domain_label = float(domain_label)
        self.img_size = img_size
        self.img_ext = img_extension
        self.mask_ext = mask_extension
        self.class_id = class_id
        self.transforms_fn = transforms_fn
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # --- Load Image & Mask ---
        idx = idx % len(self.image_files)
        img_fname = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_fname)

        # Select channels based on domain label
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if self.domain_label == 0 or self.domain_label == 2:
            image = image[..., 1]
        if image is None:
            raise RuntimeError(f"Unable to load image: {img_path}")

        mask_fname = img_fname.replace(self.img_ext, self.mask_ext)
        mask_path = os.path.join(self.masks_dir, mask_fname)
        # Load mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Unable to load mask: {mask_path}")

        # --- Initial Resize ---
        image = cv2.resize(
            image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(
            mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST
        )

        # --- Random 90° Flip ---
        if self.do_rotate and (random.random() < 0.5):
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

        # --- Small Random Rotation (-30° to +30°) ---
        if self.do_rotate:
            angle = random.uniform(-30, 30)
            h, w = image.shape[:2]
            center = (w / 2, h / 2)
            # Rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            # Compute new bounds to fit the whole image
            cos = abs(M[0, 0])
            sin = abs(M[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)
            # Adjust translation
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            # Apply warpAffine to image and mask
            image = cv2.warpAffine(
                image,
                M,
                (new_w, new_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            mask = cv2.warpAffine(
                mask,
                M,
                (new_w, new_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        # --- Final Resize Back to (img_size × img_size) ---
        image = cv2.resize(
            image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(
            mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST
        )

        # --- Contour Extraction for Bounding Boxes ---
        bin_mask = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []
        labels = []
        for cnt in contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            if w_box > 0 and h_box > 0:
                boxes.append([float(x), float(y), float(x + w_box), float(y + h_box)])
                labels.append(self.class_id)

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # --- Convert Image → Tensor [C, H, W] ---
        if image.ndim == 2:  # single‐channel
            img_tensor = torch.from_numpy(image).unsqueeze(0).float().div(255.0)
        else:
            img_tensor = (
                torch.from_numpy(image.transpose(2, 0, 1))
                .float()
                .div(255.0)
            )

        # Optional: additional torchvision‐style transforms
        if self.transforms_fn:
            img_tensor = self.transforms_fn(img_tensor)

        target = {
            "boxes": boxes,                      # [N, 4]
            "labels": labels,                    # [N]
            "domain_label": torch.tensor(
                self.domain_label, dtype=torch.float32
            ),                                   # scalar float
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        if self.use_sam:
            target["mask"] = mask

        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))