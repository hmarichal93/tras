import torch
import torchvision
from skimage.morphology import skeletonize
import cv2
import numpy as np
import segmentation_models_pytorch as smp

from shapely.geometry import LineString
from pathlib import Path
from urudendro.image import load_image, write_image
from urudendro.drawing import Drawing, Color

from deep_cstrd.dataset import create_tiles_with_labels, from_tiles_to_image, overlay_images, padding_image

from cross_section_tree_ring_detection.sampling import sampling_edges
from deep_cstrd.filter_edges import filter_edges
class segmentation_model:
    UNET = 1
    UNET_PLUS_PLUS = 2
    MASK_RCNN = 3

class RingSegmentationModel:
    def __init__(self, weights_path = "/home/henry/Documents/repo/fing/cores_tree_ring_detection/src/runs/unet_experiment/latest_model.pth" ,
                 tile_size=512, overlap=0.1, output_dir=None, model_type=segmentation_model.UNET, encoder='resnet18'):
        self.model_type = model_type
        self.tile_size = tile_size
        self.overlap = overlap
        self.output_dir = output_dir
        self.model = self.load_model(weights_path, encoder)
        self.model.eval()

    @staticmethod
    def load_architecture(model_type, encoder='resnet18', channels=3, dropout=False):
        # Load your model here
        if model_type == segmentation_model.UNET:
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights="imagenet",
                in_channels=channels,
                classes=1,
                #aux_params=dict(dropout=0.3, classes=1) if dropout else None
            )
        elif model_type == segmentation_model.UNET_PLUS_PLUS:
            model = smp.UnetPlusPlus(
                encoder_name=encoder,
                encoder_weights="imagenet",
                in_channels=channels,
                classes=1,
                #aux_params=dict(dropout=0.3, classes=1) if dropout else None
            )
        elif model_type == segmentation_model.MASK_RCNN:
            model = torchvision.models.detection.mask_rcnn.MaskRCNN(backbone="resnet50", num_classes=1, pretrained=True)

        else:
            raise ValueError("Invalid model type")

        return model

    def load_model(self, weights_path, encoder='resnet18'):

        model = self.load_architecture(self.model_type, encoder, dropout=True)

        if torch.cuda.is_available():
            model.load_state_dict(torch.load(weights_path))
            device = torch.device("cuda")
        else:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            device = torch.device("cpu")
        from torchinfo import summary
        print(summary(model, input_size=(1, 3, self.tile_size if self.tile_size>0 else 1504, self.tile_size if self.tile_size>0 else 1504)))

        model = model.to(device)
        print(f"Model is running on: {device}")
        self.device = device

        return model

    def forward(self, img, output_dir=None, batch_size=2, sigmoid=False):

        if output_dir:
            write_image(f"{output_dir}/img.png", img)
        if self.tile_size>0:
            image, _ = create_tiles_with_labels(img, img, tile_size=self.tile_size,
                                                overlap=self.overlap, inference=True)
        else:
            image = np.array([img])


        with torch.no_grad():
            image = torch.from_numpy(image).to(self.device)
            #split the image into tiles
            image = image.permute(0, 3, 1, 2).float() / 255.0  # Normalize to [0, 1]

            pred = []
            for i in range(0, image.shape[0], batch_size):
                batch = image[i:i + batch_size]
                pred += list(self.model(batch))

            pred = torch.stack(pred)
            if sigmoid:
                pred = torch.sigmoid(pred)  # Apply sigmoid to get probabilities
            pred = pred.squeeze().cpu().numpy()  # Convert to numpy array
            if self.tile_size > 0:
                pred = from_tiles_to_image(pred, self.tile_size, img, self.overlap, output_dir=output_dir, img=img)

            if output_dir:
                write_image(f"{output_dir}/pred.png", (pred * 255).astype(np.uint8))

        return pred


    def compute_connected_components_by_contour(self, skeleton, output_dir, debug=True, minimum_length=10) -> np.ndarray:

        contours, hierarchy = cv2.findContours(skeleton.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if debug:
            from urudendro.drawing import Drawing, Color
            from shapely.geometry import LineString

            img = np.zeros((skeleton.shape[0], skeleton.shape[1], 3), dtype=np.uint8)
            color = Color()

        m_ch_e = []
        contours = sorted(contours, key=lambda x: x.shape[0], reverse=True)
        for idx, c in enumerate(contours):
            c = c.squeeze()
            c_shape = c.shape[0]
            if c_shape < minimum_length:
                continue

            m_ch_e.extend(c.tolist() + [[-1,-1]])

            if debug:
                poly = LineString(c[:, ::-1])
                img = Drawing.curve(poly.coords, img, color.get_next_color())

        if debug:
            write_image(f"{output_dir}/contours.png", img)

        return np.array(m_ch_e)

    def compute_normals(self, m_ch_e, h,w):
        n = len(m_ch_e)
        Gy = np.zeros((h,w))
        Gx = np.zeros((h,w))

        for idx in range(n):
            p0 = m_ch_e[idx]
            p_m1 = m_ch_e[idx - 1] if idx > 0 else m_ch_e[idx]
            p_m2 = m_ch_e[idx + 1] if idx < n - 1 else m_ch_e[idx]

            if p0[0] == -1:
                continue

            tg = np.array(p_m2) - np.array(p_m1)
            tgnorm = tg / np.linalg.norm(tg)
            normal = np.array([-tgnorm[1], tgnorm[0]])

            i = int(p0[0])
            j = int(p0[1])
            Gy[j,i] = normal[0]
            Gx[j,i] = normal[1]


        return Gx, Gy




def rotate_image(image, center, angle=90):
    # Get image dimensions and calculate rotation matrix
    if angle == 0:
        return image
    (h, w) = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform rotation
    rotated_image = cv2.warpAffine(image.copy(), rotation_matrix, (w, h))
    return rotated_image

def remove_duplicated_elements_in_numpy_array(matrix):
    # Find unique rows and their counts
    unique_rows, indices, counts = np.unique(matrix, axis=0, return_inverse=True, return_counts=True)
    # Identify repeated rows (keep one occurrence intact)
    mask = np.zeros_like(indices, dtype=bool)
    for i, count in enumerate(counts):
        if count > 1:
            repeated_indices = np.where(indices == i)[0]  # Find all occurrences
            mask[repeated_indices[1:]] = True  # Mark all except the first one

    # Replace repeated rows with [-1, -1]
    matrix[mask] = [-1, -1]
    return matrix

def from_prediction_mask_to_curves(pred, model, output_dir=None, debug=False) -> np.ndarray:
    #skeleton = np.where(skeletonize(pred), 255, 0)
    skeleton = cv2.ximgproc.thinning(pred*255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    m_ch_e = model.compute_connected_components_by_contour(skeleton, output_dir, debug)
    m_ch_e = remove_duplicated_elements_in_numpy_array(m_ch_e.astype(int))
    return m_ch_e

def deep_contour_detector(img,
                          weights_path= "models/deep_cstrd/256_pinus_v1_1504.pth",
                          output_dir=None, cy=None, cx=None, debug=False, total_rotations=5, tile_size=0,
                          prediction_map_threshold=0.2, alpha=30, mc=2, nr=360, batch_size=1,
                          encoder='resnet18'):

    h, w = img.shape[:2]
    if h % 32 != 0 or w % 32 != 0:
        img = padding_image(img, 32)

    model = RingSegmentationModel(weights_path,encoder=encoder, tile_size=tile_size)

    if total_rotations < 1:
        total_rotations = 1

    angle_range = np.arange(0,360, 360/total_rotations).tolist()

    pred_dict = {}
    for angle in angle_range:
        output_dir_angle = Path(output_dir) / f"{angle}"
        if debug:
            output_dir_angle.mkdir(parents=True, exist_ok=True)
        output_dir_angle = None if not debug else output_dir_angle

        rot_image = rotate_image(img, (cx, cy), angle=angle)
        pred = model.forward(rot_image, output_dir=output_dir_angle, sigmoid=prediction_map_threshold != 0,
                             batch_size=batch_size)
        pred = rotate_image(pred, (cx, cy), angle=-angle)
        pred_dict[angle] = pred

    # Combine the predictions computing the average
    pred = np.zeros_like(pred_dict[angle], dtype=float)
    for angle in angle_range:
        pred += pred_dict[angle]
    pred = pred / total_rotations
    if output_dir and debug:
        draw_pred_mask(pred, img, output_dir, cx, cy)
        #invert pred to get the mask
        pred_inv = (1 - pred) * 255
        write_image(f"{output_dir}/mask.png", pred_inv)

    #binarize the mask
    pred = (pred > prediction_map_threshold).astype(np.uint8)  # Binarize the mask
    m_ch_e = from_prediction_mask_to_curves(pred, model, output_dir, debug)
    ###
    gx, gy = model.compute_normals(m_ch_e, img.shape[0], img.shape[1])
    if output_dir and debug:
        draw_normals(img, m_ch_e, gx, gy, output_dir)

    im_pre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Line 3 Edge filtering module.
    l_ch_f = filter_edges(m_ch_e, cy, cx, gy, gx, alpha, im_pre)
    # Line 4 Sampling edges.
    l_ch_s = sampling_edges(l_ch_f, cy, cx, im_pre, mc, nr, debug=debug)

    return l_ch_s

def draw_normals(img, m_ch_e, gx, gy, output_dir):
    debug_image = img.copy()

    amp = 10
    for idx in range(len(m_ch_e)):
        p0 = m_ch_e[idx]
        i = int(p0[0])
        j = int(p0[1])

        p1 = p0 + amp*np.array([gy[j,i], gx[j,i]])
        p1 = p1.astype(int)
        if p0[0] < 1 or p0[1] < 1 or p1[0] < 1 or p1[1] < 1:
            continue
        points = np.stack([p0,p1])[:,::-1]
        line = LineString(points)
        debug_image = Drawing.radii(line.coords, debug_image, color=Color.blue, thickness=1)

    for p in m_ch_e:
        if p[0] == -1:
            continue
        debug_image = Drawing.circle(debug_image, p, thickness=-1, color=Color.black, radius=1)

    write_image(f"{output_dir}/normals.png", debug_image)
    return

def draw_pred_mask(pred, img, output_dir, cx, cy):
    pred_scaled = (pred * 255).astype(np.uint8)
    #overlay the prediction on the image
    overlay = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    overlay[:,:,0] = pred_scaled
    overlay = overlay_images(img.astype(np.uint8), overlay, alpha=0.5, beta=0.5, gamma=0)

    #draw over the image the center
    overlay = Drawing.circle(overlay, (cx, cy), thickness=-1, color=Color.red, radius=5)


    write_image(f"{output_dir}/overlay.png", overlay)
    return
def test_forward(debug=False):
    weights_path = "/runs/unet_experiment/latest_model.pth"
    image_path = "/data/maestria/resultados/deep_cstrd/pinus_v1/val/images/segmented/F02d.png"
    img = load_image(image_path)
    #convert to torch tensor
    m_ch_e = deep_contour_detector(img, weights_path)

    return m_ch_e



if __name__ == "__main__":
    test_forward()