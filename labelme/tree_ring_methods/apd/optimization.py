import numpy as np
import cv2
from scipy.optimize import minimize

from lib.image import Drawing, Color, resize_image_using_pil_lib

class Optimization:
    """
    Implementation of the optimization of convex function  1/N * sum_{i=1}^N f(x_i) where f(x_i) is the functional
    cos2(theta_i) = (A.B / ||A|| ||B||)^2. Optimization is made using the gradient descent method.
    """
    def __init__(self, m_lsd, output_dir=None, img=None, weights=None, debug=False, logger=None):
        self.L_init = m_lsd
        self.img = img
        self.output_dir = output_dir
        self.debug = debug
        self.N = self.L_init.shape[0]
        self.weights_init = weights
        self.logger = logger


    @staticmethod
    def object_function(variables, coefs):

        X1, Y1, X2, Y2 = coefs[:, 0], coefs[:, 1], coefs[:, 2], coefs[:, 3]

        x,y = variables

        Xc = (X1 + X2) / 2
        Yc = (Y1 + Y2) / 2
        AA = np.array([X1 - X2, Y1 - Y2]).T
        BB = np.array([Xc - x, Yc - y]).T
        #AAnorm = cp.norm(AA, axis=1)
        #BBnorm = cp.norm(BB, axis=1)
        AA = AA / np.linalg.norm(AA, axis=1).reshape(-1, 1)
        BB = BB / np.linalg.norm(BB, axis=1).reshape(-1, 1)
        value = (AA * BB).sum(axis=1)
        value = value ** 2

        res = value.sum() / Xc.shape[0]
        return -res



    def run(self, xo, yo):
        """
        Minimize function
        """
        #loop for computing gradient descent
        H, W, _ = self.img.shape if self.img is not None else (0,0,0)


        if self.debug:
            debug_img = self.img.copy()

        results = minimize(self.object_function, [xo, yo], args=self.L_init, tol=1e-6)

        x, y = results.x
        value = -1 * results.fun

        if self.debug:
            debug_img = cv2.circle(debug_img, (np.round(x).astype(int), np.round(y).astype(int)), 2, Color.red, -1)
            cv2.imwrite(f"{self.output_dir}_op.png",
                        resize_image_using_pil_lib(debug_img, 640, 640))

        return (x, y, value )





class LeastSquaresSolution:
    """
    Implementation of the optimization of convex function  1/N * sum_{i=1}^N f(x_i) where f(x_i) is the functional
    cos2(theta_i) = (A.B / ||A|| ||B||)^2. Optimization is made using the gradient descent method.
    """
    def __init__(self, m_lsd, output_dir=None, img=None, debug=False):
        self.L_init = m_lsd
        self.img = img
        self.output_dir = output_dir
        self.debug = debug
        self.N = self.L_init.shape[0]

    @staticmethod
    def compute_line_coefficients(p1, p2):
        """
        Given two points, compute the coefficients of the line equation
        a*x+b*y+c=0
        https://bobobobo.wordpress.com/2008/01/07/solving-linear-equations-ax-by-c-0/
        """
        x1, y1 = p1.ravel()
        x2, y2 = p2.ravel()

        if x2 == x1:
            # Vertical line: x = c
            a = 1
            b = 0
            c = -x1
        elif y1 == y2:
            # Horizontal line: y = c
            a = 0
            b = 1
            c = -y1
        else:
            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1

        return a, b, c
    def run(self):
        """
        Least Squares. Minimum distance between line and dot
        ri(xi,yi) = (axi + byi + c) / sqrt(a^2 + b^2)
        """
        #loop for computing gradient descent
        H, W, _ = self.img.shape

        # Compute coefficients
        B = np.zeros((len(self.L_init), 2))
        g = np.zeros((len(self.L_init), 1))
        for idx, l in enumerate(self.L_init):
            x1, y1, x2, y2 = l.ravel()
            a, b, c = self.compute_line_coefficients(np.array((x1,y1)),np.array((x2,y2)))
            denom = np.sqrt(a ** 2 + b ** 2)
            B[idx] = np.array([a,b]) / denom
            g[idx] = -c / denom

        ## LS
        # B*[x,y]= g
        BTB = B.T.dot(B)
        try:
            invBTB = np.linalg.inv(BTB)

        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                h, w, _ = self.img.shape
                return (w // 2, h // 2)
            else:
                raise err
        x, y = invBTB.dot(B.T.dot(g))

        return (x[0], y[0])




def filter_lo_around_c(Lof, rf, ci, img_in):
    x, y = ci
    H, W = img_in.shape[:2]
    # 2.1 Select lines within region
    top_left = (x - W // rf, y - H // rf)
    bottom_right = (x + W // rf, y + H // rf)

    m_lines_within_region, _ = get_lines_idx_within_rectangular_region(img_in, Lof, None,
                                                                                           int(top_left[0]),
                                                                                           int(top_left[1]),
                                                                                           int(bottom_right[0]),
                                                                                           int(bottom_right[1]),
                                                                                           debug=False,
                                                                                           output_path=None)
    return m_lines_within_region, top_left, bottom_right

def get_lines_idx_within_rectangular_region(img_in, L, weights, top_x, top_y, bottom_x, bottom_y, debug, output_path=None):
    X1, Y1, X2, Y2 = L[:,0], L[:,1], L[:,2], L[:,3]
    idx = np.where((top_x < X1) & (top_y < Y1) & (bottom_x > X2) & (bottom_y > Y2))[0]
    l_lines_within_region = L[idx]
    #print norm l_lines_within_region
    weights_within_region = weights[idx] if weights is not None else None
    if debug:
        # draw rectangular region
        img = img_in.copy()
        img = cv2.rectangle(img, (top_x, top_y), (bottom_x, bottom_y), Color.black, 2)
        Drawing.draw_lsd_lines(l_lines_within_region, img,
                               output_path= output_path, lines_all=L)

    return l_lines_within_region,weights_within_region
