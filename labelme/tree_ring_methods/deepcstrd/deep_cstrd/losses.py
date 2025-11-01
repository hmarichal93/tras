import torch
import torch.nn as nn

class Loss:
    dice = 0
    bce = 1
class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon  # Small value to avoid division by zero

    def forward(self, y_pred, y_true):
        """
        Compute the Dice loss.
        Args:
            y_pred (Tensor): Predicted probabilities, shape (B, 1, H, W).
            y_true (Tensor): Ground truth binary masks, shape (B, 1, H, W).

        Returns:
            Tensor: Dice loss for the batch.
        """
        if type(y_pred ) == tuple:
            y_pred = y_pred[0]
        y_pred = torch.sigmoid(y_pred)  # Ensure predictions are probabilities
        y_true = y_true.float()  # Convert ground truth to float

        # Calculate intersection and union
        intersection = torch.sum(y_pred * y_true, dim=(1, 2, 3))
        union = torch.sum(y_pred, dim=(1, 2, 3)) + torch.sum(y_true, dim=(1, 2, 3))

        # Compute Dice coefficient
        dice_coeff = (2.0 * intersection + self.epsilon) / (union + self.epsilon)

        # Compute Dice loss
        dice_loss = 1.0 - dice_coeff

        # Return mean loss over the batch
        return dice_loss.mean()

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Compute the binary cross-entropy loss.
        Args:
            y_pred (Tensor): Predicted logits, shape (B, 1, H, W).
            y_true (Tensor): Ground truth binary masks, shape (B, 1, H, W).

        Returns:
            Tensor: Binary cross-entropy loss for the batch.
        """
        y_true = y_true.float()  # Convert ground truth to float

        # Compute binary cross-entropy loss
        bce_loss = nn.BCEWithLogitsLoss()(y_pred, y_true)

        return bce_loss