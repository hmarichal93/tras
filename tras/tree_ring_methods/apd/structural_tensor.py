# Extructural tensor details: https://en.wikipedia.org/wiki/Structure_tensor
# Explanation of computing structural tensor using matrix operations: https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method
# https://docs.opencv.org/4.x/d4/d70/tutorial_anisotropic_image_segmentation_by_a_gst.html
# https://www.crisluengo.net/archives/1132/ buena explicacaion
import numpy as np
import cv2
from pathlib import Path
from skimage.util.shape import view_as_windows

def StructuralTensor(img, sigma=1, window_size=5, mode = -1):
    """
    Compute structural tensor of an image.
    https://docs.opencv.org/4.x/d4/d70/tutorial_anisotropic_image_segmentation_by_a_gst.html
    @param img: input image
    @param sigma: sigma of the gaussian filter
    @param k: parameter of the Harris corner detector
    @param window_size: size of the window used to compute the structural tensor
    @param mode: if -1 is scharr if 3 is sobel
    @return: matrix with the structural tensor
    """
    # Compute the gradient of the image
    #Print input parameters
    #print(f"img.shape {img.shape} sigma {sigma} window_size {window_size} mode {mode}")
    img = img.astype(np.float32)
    img = cv2.GaussianBlur(img, (3, 3), sigma)
    img = img.astype(np.float32)

    #################################################
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=mode)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=mode)
    J11 = Ix ** 2
    J22 = Iy ** 2
    J12 = Ix * Iy

    # Compute the structural tensor
    J11 = cv2.GaussianBlur(J11, (window_size, window_size), sigma)
    J22 = cv2.GaussianBlur(J22, (window_size, window_size), sigma)
    J12 = cv2.GaussianBlur(J12, (window_size, window_size), sigma)

    #instead of gaussian blur use rectangular filter
    # J11 = cv2.boxFilter(J11, -1, (window_size, window_size))
    # J22 = cv2.boxFilter(J22, -1, (window_size, window_size))
    # J12 = cv2.boxFilter(J12, -1, (window_size, window_size))

    # Compute the eigenvalues of the structural tensor
    # eigenvalue calculation (start)
    # lambda1 = 0.5*(J11 + J22 + sqrt((J11-J22)^2 + 4*J12^2))
    # lambda2 = 0.5*(J11 + J22 - sqrt((J11-J22)^2 + 4*J12^2))
    tmp1 = J11 + J22
    tmp2 = J11 - J22
    tmp2 = cv2.multiply(tmp2, tmp2)
    tmp3 = cv2.multiply(J12, J12)
    tmp4 = np.sqrt(tmp2 + 4.0 * tmp3)
    lambda1 = 0.5 * (tmp1 + tmp4)  # biggest eigenvalue
    lambda2 = 0.5 * (tmp1 - tmp4)  # smallest eigenvalue
    # eigenvalue calculation (stop)
    # Coherency calculation (start)
    # Coherency = (lambda1 - lambda2)/(lambda1 + lambda2)) - measure of anisotropism
    # Coherency is anisotropy degree (consistency of local orientation)
    imgCoherencyOut = cv2.divide(lambda1 - lambda2, lambda1 + lambda2)
    # Coherency calculation (stop)
    # orientation angle calculation (start)
    # tan(2*Alpha) = 2*J12/(J22 - J11)
    # Alpha = 0.5 atan2(2*J12/(J22 - J11))
    imgOrientationOut = cv2.phase(J22 - J11, 2.0 * J12, angleInDegrees=False)
    imgOrientationOut = 0.5 * imgOrientationOut
    # orientation angle calculation (stop)
    return imgCoherencyOut, imgOrientationOut


def debug_image_line(width=1000, height=1000, thickness=12):
    #generate a debug image where 0s in all pixels. Draw a line that split image in two
    img = np.zeros((width,height),dtype=np.uint8) +255
    img = cv2.line(img, (0,0), (width,height), (0,0,0), thickness)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def debug_image_circle(width=1000, height=1000, thickness=12):
    #generate a debug image where 0s in all pixels. Draw a line that split image in two
    img = np.zeros((width,height),dtype=np.uint8)+255
    img = cv2.circle(img, (width//2,height//2), width//4, (0,0,0), thickness)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def sampling_structural_tensor(imgC, imgO, ST_window=3):
    #print input parameters
    #print(f"imgC.shape {imgC.shape} imgO.shape {imgO.shape} ST_window {ST_window}")
    #imgC = np.arange(7*9).reshape((7,9))
    img_h, img_w = imgC.shape
    imgO, imgC_output = imgO.copy(), np.zeros_like(imgC)

    #img_h, img_w = imgO.shape
    kernel_size = img_w // ST_window
    #check if kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size+=1


    windows_img_4d = view_as_windows(imgC, (kernel_size, kernel_size), step=kernel_size)
    #windows_img_shape = (row, col, kernel_size, kernel_size)
    windows_img_matrix = windows_img_4d.reshape(-1,kernel_size*kernel_size)


    argmax = np.argmax(windows_img_matrix, axis=1)
    windows_col = img_w // kernel_size

    windows_index = np.arange(len(argmax))
    upper_windows_row = (windows_index // windows_col) * kernel_size
    max_window_row = upper_windows_row + (argmax % kernel_size)
    max_window_col = (argmax // kernel_size) + (windows_index % windows_col) * kernel_size
    max_loc = np.vstack((max_window_row, max_window_col)).T
    imgC_output[max_loc[:,0], max_loc[:,1]] = imgC[max_loc[:,0], max_loc[:,1]]
    return imgC_output, imgO, kernel_size



def sampling_structural_tensor_matrix(imgC, imgO, kernel_size):
    #print input parameters
    #print(f"imgC.shape {imgC.shape} imgO.shape {imgO.shape} ST_window {ST_window}")
    #imgC = np.arange(7*9).reshape((7,9))
    img_h, img_w = imgC.shape
    imgO, imgC_output = imgO.copy(), np.zeros_like(imgC)



    windows_img_4d = view_as_windows(imgC, (kernel_size, kernel_size), step=kernel_size)
    #windows_img_shape = (row, col, kernel_size, kernel_size)
    windows_img_matrix = windows_img_4d.reshape(-1,kernel_size*kernel_size)


    argmax = np.argmax(windows_img_matrix, axis=1)
    windows_col = img_w // kernel_size

    windows_index = np.arange(len(argmax))
    upper_windows_row = (windows_index // windows_col) * kernel_size
    max_window_row = upper_windows_row + (argmax % kernel_size)
    max_window_col = (argmax // kernel_size) + (windows_index % windows_col) * kernel_size
    max_loc = np.vstack((max_window_row, max_window_col)).T
    imgC_output[max_loc[:,0], max_loc[:,1]] = imgC[max_loc[:,0], max_loc[:,1]]
    return imgC_output, imgO, kernel_size



def matrix_compute_local_orientation(img, W=35, Sigma=1.2, C_threshold=0.75,  ST_window=100 , debug=False, output_folder=None):
    #Structural Tensor
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).copy()
    #eq_image = equalize(gray_image)
    imgC, imgO = StructuralTensor(gray_image, window_size=W, sigma=Sigma)


    imgC, imgO, kernel_size = sampling_structural_tensor(imgC, imgO, ST_window) if ST_window > 0 else (imgC, imgO, W)

    #print(f"C_threshold {C_threshold} kernel_size {kernel_size}")
    th = np.percentile(imgC[imgC>0], 100*(1-C_threshold))

    y,x = np.where(imgC > th)
    O = imgO[y,x]
    V = np.array([np.sin(O), np.cos(O)]).T
    orientation_length = kernel_size/2
    Pc = np.array([x,y], dtype=float).T
    P1 = Pc - V * orientation_length / 2
    P2 = Pc + V * orientation_length / 2

    L = np.hstack((P1,P2))
    coherence = imgC[y,x]
    if debug:
        img_s = img.copy()
        for x1,y1,x2,y2 in L:
            p1 = np.array((x1,y1),dtype=int)
            p2 = np.array((x2,y2),dtype=int)
            img_s = cv2.line(img_s, (p1[0], p1[1]), (p2[0], p2[1]), (0, 0, 255), 1)
            #draw rectangle
            top = p1
            bottom = p2
            img_s = cv2.rectangle(img_s, (top[0], top[1]), (bottom[0], bottom[1]), (255, 0, 0), 1)

        cv2.imwrite(str(output_folder / "img_end_s.png"), img_s)

    return L, coherence
def compute_local_orientation(img, W=35, Sigma=1.2, C_threshold=0.75, debug=False):

    #Structural Tensor
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).copy()
    #eq_image = equalize(gray_image)
    imgC, imgO = StructuralTensor(gray_image, window_size=W, sigma=Sigma)

    if debug:
        img_s = img.copy()

    L = []
    coherence = []
    # for Iw in range(0, img.shape[1] - W, STRIDE):
    #     for Ih in range(0, img.shape[0] - W, STRIDE):
    for xc in range(img.shape[1]):
        for yc in range( img.shape[0] ):
            # to rad
            # yc = Ih + W//2
            # xc = Iw + W//2
            if imgC[yc, xc] < C_threshold:
                continue
            coherence.append(imgC[yc, xc])
            center_orientation = imgO[yc,xc]
            #get vector at orientation
            vector = np.array([np.sin(center_orientation), np.cos(center_orientation)])

            p1 = np.array([xc, yc],dtype=float)
            p2 = p1 + vector * W/2
            p2 = p2.astype(float)


            line = [p1[0],p1[1], p2[0], p2[1]]
            L.append(line)

    L = np.array(L)
    coherence = np.array(coherence)
    # if debug:
    #     cv2.imwrite(str(output_folder / "img_end_s.png"), img_s)


    return L, coherence

def main(filename = "./Input/F02c.png"):

    output_folder = Path("./Output") / "StructuralTensor"
    output_folder.mkdir(exist_ok=True)
    import os
    os.system(f"rm -rf {output_folder}/*")
    img = cv2.imread(filename)
    #img = debug_image_line()
    #img = debug_image_circle()


    ###############################################
    window_size = img.shape[0] // 30
    #make windows_size odd
    if window_size % 2 == 0:
        window_size += 1
    L = compute_local_orientation(img, W=window_size, debug=True)
    print(L)
    ###############################################

if __name__ == "__main__":
    filename = "./Input/D01-L1-BBF-4.jpg"
    filename = "./Input/F02c.png"
    #filename = "./Input/F02b.png"
    filename = "/data/maestria/datasets/TreeTrace_Douglas_format/logs_zoom_in/images/segmented/A11b.png"
    main(filename)