import os

import numpy as np

from deep_cstrd.dataset import load_datasets
from deep_cstrd.utils import save_batch_with_labels_as_subplots
from deep_cstrd.losses import DiceLoss, Loss
from deep_cstrd.model import segmentation_model, RingSegmentationModel

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.detection.mask_rcnn

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


def save_config(logs_dir, dataset_root, tile_size, overlap, batch_size, lr, number_of_epochs, loss, augmentation, model_type,
                encoder, debug, dropout):
    # if Path(logs_dir).exists():
    #     os.system(f"rm -r {logs_dir}")

    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    config_path = Path(logs_dir) / "config.txt"
    with open(str(config_path), "w") as f:
        f.write(f"dataset_root: {dataset_root}\n")
        f.write(f"tile_size: {tile_size}\n")
        f.write(f"overlap: {overlap}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"number_of_epochs: {number_of_epochs}\n")
        f.write(f"logs_dir: {logs_dir}\n")
        f.write(f"loss: {loss}\n")
        f.write(f"augmentation: {augmentation}\n")
        f.write(f"model_type: {model_type}\n")
        f.write(f"encoder: {encoder}\n")
        f.write(f"debug: {debug}\n")
        f.write(f"dropout: {dropout}\n")

    #load config file txt with numpy
    config = np.loadtxt(config_path, delimiter=":", dtype=str)
    print(config)


def configure_optimizer(model, lr, number_of_epochs, step_size = None, gamma = None):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # Define the learning rate scheduler
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, number_of_epochs, eta_min=lr / 100)
    return optimizer, scheduler


def forward_step(model, criterion, device, batch, debug=False):
    images, labels = batch

    # Preprocess images
    images = images.permute(0, 3, 1, 2).float() / 255.0  # Normalize to [0, 1]
    labels = labels.float().unsqueeze(1)  # Add channel dimension

    # Move data to GPU if available
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    predictions = model(images)
    # Compute loss
    loss = criterion(predictions, labels)
    if debug:
        return predictions
    return loss

def train_one_epoch(model, device, dataloader_train, optimizer, criterion, scheduler):
    model.train()
    running_loss = 0.0  # Track total loss for the epoch
    for batch_idx, batch in enumerate(dataloader_train):
        optimizer.zero_grad()  # Clear previous gradients
        # Forward pass
        loss  = forward_step(model, criterion, device, batch)
        # Backpropagation
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        # Accumulate loss
        running_loss += loss.item()

    # Step the scheduler
    scheduler.step()

    # Epoch summary
    epoch_loss = running_loss / len(dataloader_train)
    return epoch_loss

class Logger:
    def __init__(self, logs_dir):
        self.writer = SummaryWriter(log_dir=logs_dir)
        self.epoch_message = ""
        self.epoch = 0
        import numpy as np
        self.batch_idx = 4# None

    def on_training_epoch_end(self, epoch, epoch_loss, number_of_epochs):
        self.epoch_message+= f"Epoch {epoch}/{number_of_epochs} | Train Loss: {epoch_loss:.4f} | "
        self.writer.add_scalar("Train Loss/Epoch", epoch_loss, epoch)

    def on_validation_epoch_end(self, epoch, epoch_loss, lr ):
        self.epoch_message+= f"Val Loss: {epoch_loss:.4f} | lr: {lr:.6f}"
        self.writer.add_scalar("Val Loss/Epoch", epoch_loss, epoch)
        self.writer.add_scalar("lr",lr, epoch)


    def on_epoch_end(self, epoch ):
        self.epoch = epoch + 1
        print(self.epoch_message)
        self.epoch_message = ""

    def on_batch_end(self, logs, batch_idx, total_batches):
        pass

    def plot_to_image(self, figure):
        """Converts a Matplotlib figure to a NumPy array."""
        import numpy as np

        figure.canvas.draw()
        data = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
        return data

    def save_image_batch(self, dataloader_val, model, logs_dir, epoch, criterion, device, title):
        import numpy as np
        np.random.seed(4321)
        self.batch_idx = np.random.randint(0, len(dataloader_val)) if self.batch_idx is None else self.batch_idx

        batch = list(dataloader_val)[self.batch_idx]
        predictions = forward_step(model, criterion, device, batch, debug=True)
        l_fig = save_batch_with_labels_as_subplots(batch, predictions, title,
                                           output_path=None)
        for idx in range(len(l_fig)):
            fig = l_fig[idx]
            image = self.plot_to_image(fig)
            self.writer.add_image(f"val_{self.batch_idx}_{idx}", image, epoch, dataformats='HWC')

        return



def eval_one_epoch(model, device, dataloader_val, criterion):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for batch_idx, batch in enumerate(dataloader_val):
            loss = forward_step(model, criterion, device, batch)
            # Accumulate loss
            running_loss += loss.item()

    return running_loss / len(dataloader_val)

def load_model(model_type, weights_path, encoder="resnet34", channels=3, dropout=True):
    # Define the model

    model = RingSegmentationModel.load_architecture(model_type, encoder, channels=channels, dropout=dropout)

    # Ensure the model is moved to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    torch.cuda.empty_cache()

    if Path(weights_path).exists():
        model.load_state_dict(torch.load(weights_path))

    return model, device


def initializations(dataset_root= Path("/data/maestria/resultados/deep_cstrd/pinus_v1"),
                    tile_size=512, overlap=0.1, batch_size=4,
                    lr=0.001, number_of_epochs=100,
                    loss = Loss.dice, augmentation = False, model_type=segmentation_model.UNET,
                    encoder="resnet34", channels=3, thickness=3, debug=False, dropout = False,
                    min_running_loss = float("inf"), weights_path=None,
                    logs_dir="runs/unet_experiment"):
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logs_dir = Path(logs_dir) / timestamp
    logs_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = Path(dataset_root).name
    logs_name = (f"{dataset_name}_epochs_{number_of_epochs}_tile_{int(tile_size)}_batch_{batch_size}_lr_{lr}_{encoder}"
                 f"_channels_{channels}_thickness_{thickness}_loss_{loss}_model_type_{model_type}")
    if augmentation:
        logs_name += "_augmentation"
    if dropout:
        logs_name += "_dropout"
    if weights_path:
        logs_name += "_weights"
    logs_dir = str(logs_dir / logs_name)
    save_config(logs_dir, dataset_root, tile_size, overlap, batch_size, lr, number_of_epochs, loss, augmentation, model_type,
                encoder, debug, dropout)
    return logs_dir, min_running_loss, 0




def training(args):
    # Parse arguments
    dataset_root = Path(args.dataset_dir)
    logs_dir = args.logs_dir
    batch_size = args.batch_size
    tile_size = args.tile_size
    number_of_epochs = args.number_of_epochs
    overlap = args.overlap
    lr = args.lr
    loss = args.loss
    encoder = args.encoder
    channels = args.input_channels
    thickness = args.boundary_thickness
    augmentation = args.augmentation
    model_type = args.model_type
    debug = args.debug
    weights_path = args.weights_path

    # Initialize
    logs_dir, min_running_loss, best_epoch = initializations(dataset_root, tile_size, overlap, batch_size, lr,
                                                             number_of_epochs, loss, augmentation, model_type,
                                                             encoder, channels, thickness, debug, weights_path= weights_path,
                                                             logs_dir=logs_dir)

    dataloader_train, dataloader_val = load_datasets(dataset_root, tile_size, overlap, batch_size, augmentation,
                                                     thickness=thickness)

    criterion = DiceLoss() if loss == Loss.dice else nn.BCEWithLogitsLoss()

    model, device = load_model(model_type, f"{logs_dir}/best_model.pth" if weights_path is None else weights_path,
                               encoder, channels)
    from torchinfo import summary
    print(summary(model, input_size=(batch_size, channels, tile_size, tile_size)))

    optimizer, scheduler = configure_optimizer(model, lr, number_of_epochs)
    logger = Logger(logs_dir)

    for epoch in range(number_of_epochs):

        epoch_train_loss = train_one_epoch(model, device, dataloader_train, optimizer, criterion, scheduler)

        logger.on_training_epoch_end(epoch, epoch_train_loss, number_of_epochs)

        epoch_val_loss = eval_one_epoch(model, device, dataloader_val, criterion)

        logger.on_validation_epoch_end(epoch, epoch_val_loss, scheduler.get_last_lr()[0])

        logger.on_epoch_end(epoch)

        save_model = epoch_val_loss < min_running_loss
        if save_model:
            print(f"Saving model in epoch {epoch} with loss {epoch_val_loss}")
            min_running_loss = epoch_val_loss
            torch.save(model.state_dict(), f"{logs_dir}/best_model.pth")
            best_epoch = epoch
            if debug:
                logger.save_image_batch(dataloader_val, model, logs_dir, epoch, criterion, device,
                                        f"{min_running_loss:.4f}")




    #save model
    torch.save(model.state_dict(), f"{logs_dir}/latest_model.pth")
    print(f"Best model in epoch {best_epoch} with loss {min_running_loss}")
    # Close the writer after training
    logger.writer.close()

    return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a U-Net model for image segmentation')
    parser.add_argument('--dataset_dir', type=str, default="/data/maestria/resultados/deep_cstrd/pinus_v1/",
                        help='Path to the dataset directory')

    parser.add_argument('--logs_dir', type=str, default="runs/pinus_v1_40_train_12_val")
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--tile_size', type=int, default=512, help='Tile size')
    parser.add_argument('--step_size', type=int, default=20, help='Step size for the learning rate scheduler')
    parser.add_argument('--number_of_epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--overlap', type=float, default=0.1, help='Overlap between tiles')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--loss', type=int, default=0, help='Loss function. 0 dice loss, 1 BCE loss')
    parser.add_argument('--encoder', type=str, default="resnet34", help='Encoder to use')
    parser.add_argument('--boundary_thickness', type=int, default=3, help='Mask boundary thickness')
    #parser.add_argument('--encoder', type=str, default="mobilenet_v2", help='Encoder to use')
    parser.add_argument('--input_channels', type=int, default=3, help='Number of input channels')
    #load rest of parameter from config file
    parser.add_argument("--config", type=str, default="config.json", help="Path to the config file")
    parser.add_argument("--augmentation", type=bool, default=False, help="Apply augmentation to the dataset")
    parser.add_argument("--model_type", type=int, default=segmentation_model.UNET, help="Type of model to use")
    parser.add_argument("--weights_path", type=str, default=None, help="Path to the weights file")
    parser.add_argument("--debug", type=bool, default=False, help="Debug mode")
    args = parser.parse_args()

    training(dataset_root=Path(args.dataset_dir), logs_dir=args.logs_dir, augmentation= args.augmentation,
             model_type=args.model_type, debug=args.debug, batch_size=args.batch_size, tile_size=args.tile_size,
            number_of_epochs=args.number_of_epochs, overlap=args.overlap, lr=args.lr,
            encoder= args.encoder, loss=args.loss, channels=args.input_channels, thickness=args.boundary_thickness,
             weights_path=args.weights_path)

