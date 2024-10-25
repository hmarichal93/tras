import os
import numpy as np
import cv2

from abc import ABC, abstractmethod
from PIL import Image
from shapely.geometry import Polygon
from pathlib import Path

from lib.io import load_json
from lib.segmentation.u2net import U2NET



class Color:
    """BGR"""
    yellow = (0, 255, 255)
    red = (0, 0, 255)
    blue = (255, 0, 0)
    dark_yellow = (0, 204, 204)
    cyan = (255, 255, 0)
    orange = (0, 165, 255)
    purple = (255, 0, 255)
    maroon = (34, 34, 178)
    green = (0, 255, 0)
    white = (255,255,255)
    black = (0,0,0)
    gray_white = 255
    gray_black = 0

    def __init__(self):
        self.list = [Color.yellow, Color.red,Color.blue, Color.dark_yellow, Color.cyan,Color.orange,Color.purple,Color.maroon]
        self.idx = 0

    def get_next_color(self):
        self.idx = (self.idx + 1 ) % len(self.list)
        return self.list[self.idx]


class Drawing:

    @staticmethod
    def arrow(image, start_point, end_point, color, thickness=2):
        start_point = start_point.coords.xy
        end_point = end_point.coords.xy
        start = (int(start_point[0][0]), int(start_point[1][0]))[::-1]
        end = (int(end_point[0][0]), int(end_point[1][0]))[::-1]
        # Draw a line with arrow
        image = cv2.arrowedLine(image, start, end, color, thickness)
        return image
    @staticmethod
    def rectangle(image, top_left_point, bottom_right_point, color=Color.black, thickness=2):
        # Define the rectangle coordinates
        x1, y1 = top_left_point
        x2, y2 = bottom_right_point

        # Draw the rectangle on the image
        return cv2.rectangle(image.copy(), (x1, y1), (x2, y2), color,
                      thickness)  # (0, 255, 0) is the color in BGR format, and 2 is the line thickness

    @staticmethod
    def circle(image, center_coordinates,thickness=-1, color=Color.black, radius=3):
        # Draw a circle with blue line borders of thickness of 2 px
        image = cv2.circle(image, tuple(center_coordinates), radius, color, thickness)
        return image

    @staticmethod
    def put_text(text, image, org, color = (0, 0, 0), fontScale = 1 / 4):
        # font
        font = cv2.FONT_HERSHEY_DUPLEX
        # fontScale

        # Line thickness of 2 px
        thickness = 1

        # Using cv2.putText() method
        image = cv2.putText(image, text, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)

        return image

    @staticmethod
    def intersection(dot, img, color=Color.red):
        img[int(dot.y),int(dot.x),:] = color

        return img

    @staticmethod
    def curve(curva, img, color=(0, 255, 0), thickness = 1):
        y,x = curva.xy
        y = np.array(y).astype(int)
        x = np.array(x).astype(int)
        pts = np.vstack((x,y)).T
        isClosed=False
        img = cv2.polylines(img, [pts],
                              isClosed, color, thickness)

        return img

    @staticmethod
    def fill(poly, image, color, opacity=0):
        y, x = poly.xy
        y = np.array(y).astype(int)
        x = np.array(x).astype(int)
        pts = np.vstack((x, y)).T
        # Create a copy of the original image
        overlay = image.copy()

        # Draw the filled polygon on the overlay
        cv2.fillPoly(overlay, [pts], color)

        # Blend the overlay with the original image
        cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)
        return image

    @staticmethod
    def chain(chain, img, color=(0, 255, 0), thickness=5):
        y, x = chain.get_nodes_coordinates()
        pts = np.vstack((y, x)).T.astype(int)
        isClosed = False
        img = cv2.polylines(img, [pts],
                            isClosed, color, thickness)

        return img


def resize_image(image_path : Path, resize_factor : float, output_path : str = None):
    image = load_image(image_path)
    H, W = image.shape[:2]
    H_new = int(H  / resize_factor)
    W_new = int(W  / resize_factor)
    image = resize_image_using_pil_lib(image,  H_new, W_new)
    image_path = image_path if output_path is None else Path(output_path)
    write_image(str(image_path), image)
    return str(image_path)

def resize_image_using_pil_lib(im_in: np.array, height_output: object, width_output: object, keep_ratio= True) -> np.ndarray:
    """
    Resize image using PIL library.
    @param im_in: input image
    @param height_output: output image height_output
    @param width_output: output image width_output
    @return: matrix with the resized image
    """

    pil_img = Image.fromarray(im_in)
    # Image.ANTIALIAS is deprecated, PIL recommends using Reampling.LANCZOS
    #flag = Image.ANTIALIAS
    flag = Image.Resampling.LANCZOS
    if keep_ratio:
        aspect_ratio = pil_img.height / pil_img.width
        if pil_img.width > pil_img.height:
            height_output = int(width_output * aspect_ratio)
        else:
            width_output = int(height_output / aspect_ratio)

    pil_img = pil_img.resize((width_output, height_output), flag)
    im_r = np.array(pil_img)
    return im_r


def load_image(image_path):
    """Load image from path"""
    return cv2.imread(str(image_path),cv2.IMREAD_UNCHANGED)

def write_image(image_path, image):
    """Write image to path"""
    cv2.imwrite(str(image_path), image)
    return


################salient object
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image



# Función para cargar el modelo preentrenado
def load_model(model_path='u2net.pth'):
    model = U2NET()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocesamiento de la imagen de entrada (sin cambiar resolución)
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # Guardar el tamaño original
    transform = transforms.Compose([
        transforms.Resize((320, 320)),  # Mantener 320x320 solo para el modelo
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_resized = transform(image).unsqueeze(0)  # Añadir batch
    return image_resized, original_size, image  # Retorna también la imagen original para usarla luego

# Función para procesar la imagen con el modelo U2-Net
def salient_object_detection(model, image_tensor):
    with torch.no_grad():
        d1, _, _, _, _, _, _ = model(image_tensor)
        pred = d1[:, 0, :, :]
        pred = F.upsample(pred.unsqueeze(0), size=(320, 320), mode='bilinear', align_corners=False)
        pred = pred.squeeze().cpu().numpy()
        pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))  # Normalización
    return pred

# Postprocesamiento: Redimensionar la máscara a la resolución original
def apply_mask(image, mask, original_size):
    mask = cv2.resize(mask, original_size)  # Ajustar la máscara a la resolución original
    mask = np.expand_dims(mask, axis=2)
    image = np.array(image) * mask  # Aplicar la máscara a la imagen original
    #change background to white
    #convert mask to gray scale
    mask = (mask * 255).astype(np.uint8)
    y, x, _ = np.where(mask == 0)
    image[y, x] = 255
    return image, mask

# Guardar la imagen final sin el objeto saliente
def save_image(output_path, result_image):
    #conver to BGR
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_image)

# Pipeline para eliminar el objeto saliente manteniendo la resolución original
def remove_salient_object(image_path, output_path, model_path='./models/segmentation/u2net.pth'):
    model = load_model(model_path)
    image_tensor, original_size, original_image = preprocess_image(image_path)
    mask = salient_object_detection(model, image_tensor)
    result_image, mask_original_dim = apply_mask(original_image, mask, original_size)
    save_image(output_path, result_image)
    return mask_original_dim
