import os, glob, time
import numpy as np
from PIL import Image

os.environ['TORCH_HOME'] = '/scratch/karthiavv/thesis_project/torch_cache'
os.environ['HF_HOME'] = '/scratch/karthiavv/thesis_project/hf_cache'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from transformers import SegformerForSemanticSegmentation
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

CLASS_NAMES = ["Ignore", "Background", "Building", "Road", "Water", "Barren", "Forest", "Agriculture"]
NUM_CLASSES = 8

LOVEDA_COLORS = np.array([
    [255, 255, 255], [0, 0, 0], [255, 0, 0], [255, 255, 0], 
    [0, 0, 255], [159, 129, 183], [0, 255, 0], [255, 195, 128]
], dtype=np.uint8)

def mask_to_color(mask_np):
    mask_np = np.clip(mask_np, 0, len(LOVEDA_COLORS) - 1)
    return Image.fromarray(LOVEDA_COLORS[mask_np])

def denormalize_image(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return Image.fromarray((tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=0):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float()
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1.0)
        return loss.mean()

class LoveDADataset(Dataset):
    def __init__(self, split_dir):
        super().__init__()
        raw_images = sorted(glob.glob(os.path.join(split_dir, '*', 'images_png', '*.png')))
        self.images = []
        self.masks = []
        for img_path in raw_images:
            mask_path = img_path.replace('images_png', 'masks_png')
            if os.path.exists(mask_path):
                self.images.append(img_path)
                self.masks.append(mask_path)

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        filename = os.path.splitext(os.path.basename(self.images[idx]))[0]
        image = Image.open(self.images[idx]).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
        mask = Image.open(self.masks[idx]).resize((512, 512), Image.Resampling.NEAREST)
        
        image_tensor = TF.to_tensor(image)
        image_norm = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask_tensor = torch.as_tensor(TF.pil_to_tensor(mask), dtype=torch.long).squeeze(0)
        return image_norm, mask_tensor, filename

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dir = '/scratch/karthiavv/thesis_project/LoveDA_Data/Train'
    val_dir   = '/scratch/karthiavv/thesis_project/LoveDA_Data/Val'
    save_dir  = '/scratch/karthiavv/thesis_project/outputs_segformer_512_1gpu'

    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'ground_truth'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)

    epochs = 50
    batch_size = 8 
    base_lr = 6e-5
    warmup_epochs = 5

    print(f"=== SegFormer-B2 | 512x512 | 1 GPU ===")
    
    train_dataset = LoveDADataset(train_dir)
    val_dataset = LoveDADataset(val_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    criterion = FocalLoss(gamma=2.0, ignore_index=0).to(device)
    
    metric_iou = MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=0, average='none').to(device)
    metric_acc = MulticlassAccuracy(num_classes=NUM_CLASSES, ignore_index=0, average='none').to(device)

    for epoch in range(epochs):
        epoch_start = time.time()

        if epoch < warmup_epochs: 
            lr_scale = (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            lr_scale = 0.5 * (1.0 + np.cos(np.pi * progress))
            
        for param_group in optimizer.param_groups: 
            param_group['lr'] = base_lr * lr_scale

        current_lr = optimizer.param_groups[0]['lr']
        model.train()
        running_loss = 0.0

        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            logits = model(pixel_values=images).logits
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        model.eval()
        metric_iou.reset()
        metric_acc.reset()
        is_final_epoch = (epoch == epochs - 1)

        with torch.no_grad():
            for images, masks, filenames in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                logits = model(pixel_values=images).logits
                logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                preds = torch.argmax(logits, dim=1)
                
                metric_iou.update(preds, masks)
                metric_acc.update(preds, masks)

                if is_final_epoch:
                    for i in range(images.size(0)):
                        name = filenames[i]
                        orig_img = denormalize_image(images[i])
                        gt_colored = mask_to_color(masks[i].cpu().numpy())
                        pred_colored = mask_to_color(preds[i].cpu().numpy())

                        orig_img.save(os.path.join(save_dir, 'images', f"{name}.png"))
                        gt_colored.save(os.path.join(save_dir, 'ground_truth', f"{name}_gt.png"))
                        pred_colored.save(os.path.join(save_dir, 'predictions', f"{name}_pred.png"))

        per_class_iou = metric_iou.compute()
        per_class_acc = metric_acc.compute()

        val_miou = torch.nanmean(per_class_iou[1:]).item()
        epoch_sec = time.time() - epoch_start
        
        print(f"\n-------------------------------------------------------------------------")
        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | LR: {current_lr:.6f} | Time: {epoch_sec:.2f}s | Loss: {avg_loss:.4f} | mIoU: {val_miou:.4f}")
        print(f"-------------------------------------------------------------------------")
        print(f"{'Class':<15} | {'IoU':<10} | {'Accuracy':<10}")
        for cls_idx in range(1, NUM_CLASSES):
            print(f"{CLASS_NAMES[cls_idx]:<15} | {per_class_iou[cls_idx].item():<10.4f} | {per_class_acc[cls_idx].item():<10.4f}")
        print(f"-------------------------------------------------------------------------\n")

        if is_final_epoch:
            torch.save(model.state_dict(), f"{save_dir}/segformer_512_1gpu_final.pth")
            print(f"Final epoch complete. Images and weights saved in: {save_dir}")

if __name__ == '__main__':
    main()