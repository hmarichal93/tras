import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from urudendro.image import write_image

def save_batch_with_labels_as_subplots(batch,  predictions,title, output_path="batch_predictions_with_labels.png",
                                       threshold=0.5, batch_size=2):
    """
    Save a batch of images, labels, and predictions as a single subplot.
    Args:
        images (Tensor): Batch of input images (B, C, H, W).
        labels (Tensor): Batch of ground truth masks (B, 1, H, W).
        predictions (Tensor): Batch of predicted masks (B, 1, H, W).
        output_path (str): Path to save the output figure.
        threshold (float): Threshold for binary masks.
    """
    # Ensure predictions are probabilities and convert to binary masks
    probabilities = torch.sigmoid(predictions)  # Convert logits to probabilities
    binary_masks = probabilities > threshold    # Apply threshold
    images, labels = batch

    images_size = images.size(0)
    total_figures = np.ceil(images_size / 4).astype(int)
    l_figs = []
    for idx_fig in range(total_figures):
        fig, axes = plt.subplots(4, 3, figsize=(15, 5 * batch_size))
        l_figs.append(fig)
        fig.suptitle(title)
        if images_size == 1:
            return

        for i in range(4):
            # Convert image, label, and mask to NumPy
            image_np = images[idx_fig + i].cpu().numpy().astype(np.uint8)#.transpose(1, 2, 0)  # Convert to HWC
            #image_np = (image_np * 255).astype(np.uint8)           # Rescale to [0, 255]
            #convert to RGB
            #write_image(f"image_{idx_fig}_{i}.png", image_np)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            label_np = labels[idx_fig + i].cpu().numpy().squeeze()           # Squeeze channel dimension
            label_np = np.clip((label_np * 255),0,255).astype(np.uint8)           # Rescale to [0, 255]

            mask_np = binary_masks[idx_fig + i].cpu().numpy().squeeze()      # Squeeze channel dimension

            # Plot original image
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title(f"Image {i}")
            axes[i, 0].axis("off")

            # Plot ground truth label
            axes[i, 1].imshow(image_np)
            axes[i, 1].imshow(label_np, cmap="jet", alpha=0.5)  # Display label as grayscale
            axes[i, 1].set_title(f"Label {i}")
            axes[i, 1].axis("off")

            # Plot predicted mask
            axes[i, 2].imshow(image_np)
            axes[i, 2].imshow(mask_np, cmap="jet", alpha=0.5)  # Overlay mask
            axes[i, 2].set_title(f"Prediction {i}")
            axes[i, 2].axis("off")

        # Adjust layout and save
        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    return l_figs
