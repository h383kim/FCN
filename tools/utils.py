import torch
import random
import matplotlib.pyplot as plt
from data.CamVid_Dataset import *

def label_to_rgb_tensor(label_tensor: torch.Tensor) -> torch.Tensor:
    height, width = label_tensor.shape
    rgb_image = torch.zeros(3, height, width, dtype=torch.uint8)

    for label, rgb in label2RGB_dict.items():
        mask = (label_tensor == label)
        rgb_image[0][mask] = rgb[0]  # Red
        rgb_image[1][mask] = rgb[1]  # Green
        rgb_image[2][mask] = rgb[2]  # Blue

    return rgb_image

def visualize_segmentation(model, val_loader, device, epoch):
    model.eval()
    batch_idx = random.randint(0, len(val_loader) - 1)
    images, labels = list(val_loader)[batch_idx]
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

    img_idx = random.randint(0, len(images) - 1)
    img = images[img_idx].cpu().numpy().transpose(1, 2, 0)
    label = labels[img_idx].cpu().numpy()
    pred = preds[img_idx].cpu().numpy()
    pred_rgb = label_to_rgb_tensor(pred).cpu().numpy().transpose(1, 2, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    
    axes[1].imshow(label_to_rgb_tensor(label).permute(1, 2, 0).cpu().numpy())
    axes[1].set_title('Ground Truth')

    axes[2].imshow(pred_rgb)
    axes[2].set_title('Predicted Mask')

    # 이미지 파일 저장
    output_path = f'/kaggle/working/segmentation_epoch_{epoch}.png'
    plt.savefig(output_path)
    plt.close()

    print(f"Segmentation visualization saved at {output_path}")