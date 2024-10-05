"""
Copyright (c) 2023 Author(s) Henry Marichal (hmarichal93@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
import argparse
from shapely.geometry import Polygon, LineString
from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt

from lib.io import load_image, load_json
from lib.sampling import build_rays, compute_intersection, draw_ray_curve_and_intersections
import lib.chain as ch
import lib.drawing as dr

FP_ID = -1


class Polygon_node(Polygon):
    def __init__(self, node_list):
        self.angles = [node.angle for node in node_list]
        self.node_list = node_list
        super().__init__([[node.y, node.x] for node in node_list])


class InfluenceArea:
    def __init__(self, gt_file, dt_file, img_filename, output_dir, threshold, cy, cx, Nr=360):
        # 1.0 generate output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 2.0 set parameters
        self.threshold = threshold
        self.Nr = Nr
        self.center = [cy, cx]
        # 3.0 load image
        self.img = load_image(img_filename)
        # 4.0 sampling detection and groud truth rings by Nr rays.
        height, width, _ = self.img.shape
        l_rays = build_rays(self.Nr, height, width, self.center)
        self.dt_poly = self.get_sampled_polygon_rings(dt_file, l_rays, self.center)
        self.dt_poly.sort(key=lambda x: x.area)
        self.gt_poly = self.get_sampled_polygon_rings(gt_file, l_rays, self.center)
        self.gt_poly.sort(key=lambda x: x.area)
        # 5.0 draw rays and rings
        self.draw_ray_and_dt_and_gt(l_rays, self.gt_poly, self.dt_poly, self.img.copy(),
                                    f'{self.output_dir}/dots_curve_and_rays.png')




    def draw_ray_and_dt_and_gt(self, rays_list, l_gt_poly, l_dt_poly, img_draw, filename):
        img_draw = cv.cvtColor(img_draw, cv.COLOR_BGR2RGB)
        for ray in rays_list:
            img_draw = dr.Drawing.radii(ray, img_draw)

        for curve in l_gt_poly:
            img_draw = dr.Drawing.curve(curve.exterior, img_draw, color=dr.Color.green)

        for curve in l_dt_poly:
            img_draw = dr.Drawing.curve(curve.exterior, img_draw, color=dr.Color.red)

        cv.imwrite(filename, img_draw)

    def get_sampled_polygon_rings(self, ring_filename, l_rays, center):
        l_poly = self.load_ring_stimation(ring_filename)
        l_poly_sampled = self.sampling_rings(l_poly, l_rays, center)
        return l_poly_sampled

    def sampling_rings(self, l_poly, l_rays, center):
        l_poly_samples = []
        cy, cx = center
        for poly in l_poly:
            sampled_poly = self._sampling_poly(poly, cy, cx, l_rays)
            if sampled_poly is None:
                continue
            l_poly_samples.append(sampled_poly)

        return l_poly_samples

    @staticmethod
    def load_ring_stimation(path):
        try:
            json_content = load_json(path)
            l_rings = []
            for ring in json_content['shapes']:
                l_rings.append(Polygon(np.array(ring['points'])[:, [1, 0]].tolist()))

        except FileNotFoundError:
            l_rings = []

        return l_rings

    def _convert_poly_dict_to_poly_list(self, poly_d):
        poly_list = []
        for key in poly_d.keys():
            poly_list.append(poly_d[key])

        return poly_list

    def _add_poly_to_img(self, img, poly, color, thickness=1):
        isClosed = True
        y, x = poly.exterior.coords.xy
        pts = np.vstack((x, y)).T.astype(np.int32)
        pts = pts.reshape((-1, 1, 2))

        return cv.polylines(img, [pts], isClosed, color, thickness)

    @staticmethod
    def _sampling_poly(poly, cy, cx, l_rays):
        l_curve_nodes = compute_intersection(l_rays, poly, -1, [cy, cx])
        if l_curve_nodes is None:
            return None
        l_curve_nodes.sort(key=lambda x: x.angle)

        return Polygon_node(l_curve_nodes)

    def _plot_gt_and_dt_polys(self, img, gt, dt, n=1, title=None):
        img_aux = img.copy()
        # gt
        for poly in gt:
            img_aux = self._add_poly_to_img(img_aux, poly, color=(0, 255, 0), thickness=n)

        # dt
        for poly in dt:
            img_aux = self._add_poly_to_img(img_aux, poly, color=(255, 0, 0), thickness=n)

        plt.figure(figsize=(10, 10));
        plt.imshow(img_aux);
        plt.axis('off');
        if title is not None:
            plt.title(title)
        plt.savefig(f'{self.output_dir}/dt_and_gt.png')
        # plt.show();
        plt.close()


    def get_x_and_y_coordinates(self, poly: Polygon_node):
        # if type(poly) is Polygon:
        #     y, x  = poly.exterior.coords.xy
        #     return x, y
        x = [int(node.x) for node in poly.node_list]
        y = [int(node.y) for node in poly.node_list]
        return x, y
    def extract_poly_coordinates(self, poly):
        x, y = self.get_x_and_y_coordinates(poly)
        pts = np.vstack((y, x)).T.astype(np.int32)
        return pts

    def generate_radial_error_heat_map(self):
        """
        Generate a heat map of radial difference error between dt and gt
        @return:
        """
        self.color_map = np.zeros((len(self.gt_poly), self.Nr)) + np.nan
        for idx, dt in enumerate(self.dt_poly):
            dt_nodes = dt.node_list#self.extract_poly_coordinates(dt)
            if len(self.dt_and_gt_assignation) - 1 < idx:
                continue

            gt_idx = self.dt_and_gt_assignation[idx]
            if gt_idx == FP_ID:
                continue

            gt_nodes = self.gt_poly[gt_idx].node_list#.extract_poly_coordinates(self.gt_poly[gt_idx])

            for ray_direction_i in range(self.Nr):
                dti_node = ch.get_node_from_list_by_angle(dt_nodes, ray_direction_i)#self.get_dot_by_ray_direction_index(ray_direction_i, dt_dots)
                gti_node =ch.get_node_from_list_by_angle(gt_nodes, ray_direction_i)# self.get_dot_by_ray_direction_index(ray_direction_i, gt_dots)
                radial_distance_dti = dti_node.radial_distance#self.compute_radial_distance(dti_dot)
                radial_distance_gti = gti_node.radial_distance#self.compute_radial_distance(gti_dot)
                radial_difference = radial_distance_dti - radial_distance_gti

                self.color_map[gt_idx, int(ray_direction_i)] = radial_difference

        self.plot_color_map(self.color_map)
        return 0

    def compute_rmse_between_dt_and_gt(self, dt_poly, gt_poly):
        """
        Compute the RMSE between the dt and gt polygons. Each polygon has Nr points.
        @param dt_poly: detection ring polygon
        @param gt_poly: ground truth ring polygon
        @return: rmse error
        """
        error = []
        l_dt_dot = self.extract_poly_coordinates(dt_poly)
        l_gt_dot = self.extract_poly_coordinates(gt_poly)
        for ray_direction_idx in range(self.Nr):
            dt_dot = self.get_dot_by_ray_direction_index(ray_direction_idx, l_dt_dot)
            gt_dot = self.get_dot_by_ray_direction_index(ray_direction_idx, l_gt_dot)
            radial_distance_dt = self.compute_radial_distance(dt_dot)
            radial_distance_gt = self.compute_radial_distance(gt_dot)
            radial_difference = radial_distance_dt - radial_distance_gt
            error.append(radial_difference ** 2)

        return np.sqrt(np.mean(error))

    def plot_color_map(self, polar_heat_map):
        # polar
        # https://stackoverflow.com/questions/36513312/polar-heatmaps-in-python

        rad = np.linspace(0, len(self.gt_poly), len(self.gt_poly) + 1)
        theta = np.linspace(0, 2 * np.pi, self.Nr)
        th, r = np.meshgrid(theta, rad)

        cmaps = ['Spectral']
        for cmap_label in cmaps:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='polar')
            pcm = ax.pcolormesh(th, r, polar_heat_map, cmap=plt.get_cmap(cmap_label))
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_theta_zero_location('S')
            plt.title(f'Heat map radial error between dt and gt')
            fig.colorbar(pcm, ax=ax, orientation="vertical")
            fig.savefig(f"{self.output_dir}/heat_map_{cmap_label}.png")

            plt.close()

    def compute_radial_distance(self, pt):
        return ch.euclidean_distance([pt[1], pt[0]], self.center)

    def compute_rmse_global(self):
        l_rmse = [None] * len(self.gt_poly)
        for idx, dt in enumerate(self.dt_poly):
            if len(self.dt_and_gt_assignation) - 1 < idx:
                continue

            gt_idx = self.dt_and_gt_assignation[idx]
            if gt_idx == FP_ID:
                continue

            gt = self.gt_poly[gt_idx]
            rmse = self.compute_rmse_between_dt_and_gt(dt, gt)
            l_rmse[gt_idx] = rmse

        mean_rmse = np.mean([rmse for rmse in l_rmse if rmse is not None])
        l_rmse = np.array(l_rmse)
        self.plot_rmse_per_ring(np.where(l_rmse == None, 0, l_rmse), mean_rmse)
        return mean_rmse

    def plot_rmse_per_ring(self, l_rmse, overal_rmse):
        plt.figure()
        plt.bar(np.arange(0, len(l_rmse)), l_rmse)
        plt.title(f"RMSE global={overal_rmse:.3f}")
        plt.xlabel('Number Ring')
        plt.ylabel('RMSE (per gt)')
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/rmse.png")
        # plt.show()
        plt.close()

    def get_dot_by_ray_direction_index(self, alpha, pts):
        return pts[int(alpha)].reshape(1, 2)[0]

    def mean_interpolation(self, dot1, dot2):
        """
        Compute mean interpolation between two dots
        @param dot1: pixel dot 1
        @param dot2: pixel dot 2
        @return: mean pixel dot
        """
        return (0.5 * (dot1 + dot2)).astype(int).tolist()

    def mirror_interpolation(self, dot1, dot2):
        return (dot1 + (dot1 - dot2)).astype(int).tolist()

    def generate_new_poly(self, pol1, pol2, type_interpolation='mean'):
        """
        Generate new polygon from two polygons. If type_interpolation is mean, compute mean interpolation between two dots.
        In other case, compute mirror interpolation between two dots. Mirror interpolation means that a new external
        polygon is generated given the pol1 and pol2. When type of interpolation is mirror, pol2 is the internal polygon
         and pol1 is the external polygon. A new polygon is generated with the same number of dots that pol1 and pol2. In addition
         this new polygon is generated with the same center that pol1 and pol2 and surrounding pol1 (external).
        @param pol1: pol1
        @param pol2: pol2
        @param type_interpolation: mean or mirror
        @return: new polygon
        """
        pol1_dots = self.extract_poly_coordinates(pol1)
        pol2_dots = self.extract_poly_coordinates(pol2)
        dots_new_poly = []
        for ray_direction_idx in range(self.Nr):
            angle_dot_1 = self.get_dot_by_ray_direction_index(ray_direction_idx, pol1_dots)
            angle_dot_2 = self.get_dot_by_ray_direction_index(ray_direction_idx, pol2_dots)
            new_dot = self.mean_interpolation(angle_dot_1, angle_dot_2) if type_interpolation in 'mean' else \
                self.mirror_interpolation(angle_dot_1, angle_dot_2)
            # dot is an array [x, y]. We neet to convert it to node object
            new_node = ch.Node(x = int(new_dot[1]), y = int(new_dot[0]), angle=ray_direction_idx,
                            radial_distance = self.compute_radial_distance(new_dot), chain_id = -1)
            dots_new_poly.append(new_node)
        return Polygon_node(dots_new_poly)

    def _build_influence_area(self, img, l_gt_poly):
        """
        Compute influence matrix. Each pixel has a value that indicates the gt_poly that influences it the most.
        @param img: image matrix
        @param l_gt_poly: ground truth polygon list
        @return: influence matrix
        """
        influence_matrix = np.zeros((img.shape[1], img.shape[0])) - 1
        l_gt_poly.sort(key=lambda x: x.area)

        M, N, _ = self.img.shape
        i = 0
        for gt_i in l_gt_poly:
            gt_i_plus_1 = l_gt_poly[i + 1] if i < len(l_gt_poly) - 1 else None
            gt_i_minus_1 = l_gt_poly[i - 1] if i > 0 else None

            Cm = self.generate_new_poly(gt_i, gt_i_minus_1) if gt_i_minus_1 is not None else None
            CM = self.generate_new_poly(gt_i, gt_i_plus_1) if gt_i_plus_1 is not None else \
                self.generate_new_poly(gt_i, Cm, type_interpolation='mirror')

            if Cm is None:
                # 1. First ring
                contours = self.extract_poly_coordinates(CM)
                mask = np.zeros((N, M))
                cv.fillPoly(mask, pts=[contours], color=(255))

            else:
                contours = self.extract_poly_coordinates(CM)
                mask_M = np.zeros((N, M))
                cv.fillPoly(mask_M, pts=[contours], color=(255))

                contours = self.extract_poly_coordinates(Cm)
                mask_m = np.zeros((N, M))
                cv.fillPoly(mask_m, pts=[contours], color=(255))
                mask = mask_M - mask_m

            influence_matrix[mask > 0] = i
            i += 1

        return influence_matrix

    def true_positive(self):
        return np.where(np.array(self.dt_and_gt_assignation) > -1)[0].shape[0]

    def false_negative(self):
        dt_asigned = self.true_positive()
        return len(self.gt_poly) - dt_asigned

    def false_positive(self):
        return np.where(np.array(self.dt_and_gt_assignation) == FP_ID)[0].shape[0]

    def true_negative(self):
        # It does not make sense in this case
        return 0

    def precision(self, TP, FP, TN, FN):
        return TP / (TP + FP) if TP + FP > 0 else 0

    def recall(self, TP, FP, TN, FN):
        return TP / (TP + FN) if TP + FN > 0 else 0

    def fscore(self, TP, FP, TN, FN):
        P = self.precision(TP, FP, TN, FN)
        R = self.recall(TP, FP, TN, FN)
        return 2 * P * R / (P + R) if P + R > 0 else 0

    def _assign_gt_to_dt(self, influence_matrix, l_dt_poly):
        """
        Assign each detection polygon to a ground truth polygon. The assignment is done by computing the number of pixels
        inside the ground truth influence area. The detection polygon is assigned to the ground truth polygon with lower
        RMSE error
        @param influence_matrix: ground truth influcen matrix
        @param l_dt_poly: detection polygon list
        @return: vector detection assignation and percentage of detection point inside the ground truth influence area.
        """
        threshold = self.threshold
        dt_and_gt_assignation = []
        accuracy_percentage = []
        l_dt_poly.sort(key=lambda x: x.area)

        for poly in l_dt_poly:
            # 1.0 extract detection poly coordinates
            x, y = self.get_x_and_y_coordinates(poly)
            #y, x = poly.exterior.coords.xy
            #x = np.array(x).astype(int)
            #y = np.array(y).astype(int)

            # 2.0 extract influence matrix values for detection poly coordinates
            gts = influence_matrix[x, y].astype(int)

            # 3.0 compute the number of detection pixels that are influenced by each gt_poly
            mask_no_background = np.where(gts >= 0)[0]
            counts = np.bincount(gts[mask_no_background])

            if len(counts) == 0:
                # 3.1 no gt_poly influences the detection poly
                dt_and_gt_assignation.append(FP_ID)
                continue

            # 4.0 Find the gt_poly that influences the most pixels of the detection poly
            error_vector_between_detection_and_gt_lists = [self.compute_rmse_between_dt_and_gt(poly, gt_poly) for
                                                           gt_poly in self.gt_poly]
            gt_label = np.argmin(error_vector_between_detection_and_gt_lists)
            if gt_label in dt_and_gt_assignation:
                # 4.1 the gt_poly has already been assigned to another detection poly.
                # Check if the current detection poly has a lower rmse error
                rmse_current = self.compute_rmse_between_dt_and_gt(poly, self.gt_poly[gt_label])
                id_dt_former = np.where(dt_and_gt_assignation == gt_label)[0][0]
                rmse_former = self.compute_rmse_between_dt_and_gt(self.dt_poly[id_dt_former], self.gt_poly[gt_label])
                if rmse_current < rmse_former:
                    # 4.1.1 the current detection poly has a lower rmse error than the former one. Change former assignation
                    # to be a false positive
                    dt_and_gt_assignation[id_dt_former] = FP_ID
                    try:
                        accuracy_percentage[id_dt_former] = FP_ID
                    except Exception as e:
                        continue
                else:
                    # 4.1.2 the current detection poly has a higher rmse error than the former one.
                    # Assign it as a false positive
                    dt_and_gt_assignation.append(FP_ID)
                    accuracy_percentage.append(FP_ID)
                    continue

            # 5.0 assign the gt_poly to the detection poly
            dt_and_gt_assignation.append(gt_label)
            # 6.0 compute the accuracy percentage of the detection poly.
            accuracy_percentage.append(counts[gt_label] / self.Nr)

        # 7.0 assign as false positive the detection polys that have an accuracy percentage lower than the threshold
        no_asigned_idx = np.where(np.array(accuracy_percentage) < threshold)[0].astype(int)
        dt_and_gt_assignation = np.array(dt_and_gt_assignation)
        dt_and_gt_assignation[no_asigned_idx] = FP_ID
        accuracy_percentage = np.array(accuracy_percentage)
        accuracy_percentage[no_asigned_idx] = FP_ID
        dt_and_gt_assignation = dt_and_gt_assignation.tolist()
        accuracy_percentage = accuracy_percentage.tolist()

        return dt_and_gt_assignation, accuracy_percentage

    def _plot_assignation_between_gt_and_dt(self):
        import itertools
        import matplotlib.cm as cm
        M, N, _ = self.img.shape
        plt.figure(figsize=(10, 10))
        plt.imshow(np.zeros((M, N)), cmap='gray')

        espaciado_color = 10
        index = np.linspace(0, 1, espaciado_color)
        lista_colores = cm.rainbow(index)
        index_order = np.arange(espaciado_color)
        np.random.shuffle(index_order)
        lista_colores = lista_colores[index_order]
        colors = itertools.cycle(lista_colores)
        for idx, dt in enumerate(self.dt_poly):
            c = next(colors)
            y, x = dt.exterior.coords.xy
            plt.plot(x, y, color=c)
            # plt.scatter(x,y,s=1)
            if len(self.dt_and_gt_assignation) - 1 < idx:
                continue
            gt_idx = self.dt_and_gt_assignation[idx]
            if gt_idx == FP_ID:
                plt.plot(x, y, color='w')
                continue
            y, x = self.gt_poly[gt_idx].exterior.coords.xy
            plt.plot(x, y, color=c)
            # plt.scatter( x,y, s=1)

        plt.axis('off')
        plt.savefig(f'{self.output_dir}/assigned_dt_gt.png')
        plt.close()

    def compute_indicators(self):
        influence_matrix = self._build_influence_area(self.img, self.gt_poly)
        self._plot_influece_area(influence_matrix, self.gt_poly)
        self.dt_and_gt_assignation, self.accuracy_percentage = self._assign_gt_to_dt(influence_matrix, self.dt_poly)
        self._plot_assignation_between_gt_and_dt()

        TP = self.true_positive()
        FP = self.false_positive()
        TN = self.true_negative()
        FN = self.false_negative()
        self._plot_gt_and_dt_polys(self.img, self.gt_poly, self.dt_poly, n=3, title=f"TP={TP} FP={FP} TN={TN} FN={FN}")
        return TP, FP, TN, FN

    def _plot_influece_area(self, matriz, list_gt_poly):
        import itertools
        import matplotlib.cm as cm
        M, N, _ = self.img.shape
        plt.figure(figsize=(15, 15))
        plt.imshow(np.zeros((M, N)), cmap='gray')
        espaciado_color = 10
        index = np.linspace(0, 1, espaciado_color)
        lista_colores = cm.rainbow(index)
        index_order = np.arange(espaciado_color)
        np.random.shuffle(index_order)
        lista_colores = lista_colores[index_order]
        colors = itertools.cycle(lista_colores)
        regiones_id = list(np.unique(matriz).astype(int))
        # regiones_id.remove(-1)

        for region in regiones_id:

            x, y = np.where(matriz == region)
            if region == -1:
                plt.scatter(x, y, s=1, c='w')
            else:
                plt.scatter(x, y, s=1, color=next(colors))

        for poly in list_gt_poly:
            y, x = self.get_x_and_y_coordinates(poly)#poly.exterior.coords.xy
            plt.plot(y, x, 'k')

        plt.axis('off')
        plt.savefig(f'{self.output_dir}/influence_area.png')
        plt.close()


def main(dt_file, gt_file, img_filename, output_dir, threshold, cx, cy):
    """
    Compute the influence area metric between the  ground truth and detection rings. The metric is computed as follows:
    0.0 Sampling the ground truth and detection rings to the same number of nodes. By default, Nr=360
    1.0 Compute the influence area of each ground truth ring.
    2.0 Assign the detection rings to the ground truth rings. Detection ring is assigned to the
    closest ground truth ring following metric: sqrt(1/Nr * sum((dti-gti)^2)) where Nr is the number of rays. Where dti is
    the radial distance of the i node of detection ring and gti is the radial distance of the i node of ground truth ring.
    2.1 If more than one detection ring is assigned to the same ground truth ring, the detection ring with the lowest RMSE
    is kept and the rest are considered as false positive.
    3.0 Get total nodes of the detection inside the assigned ground truth influence area (dt_total).
    If dt_total/Nr is lower than threshold, the detection ring is considered as false positive.
    4.0 Ground truth with no assigned detection ring are considered as false negative.
    5.0 Compute the precision, recall, F-score metrics and mean RMSE of the assigned detection rings.

    @param dt_file: detection filename
    @param gt_file: groud truth filename
    @param img_filename: image filename
    @param output_dir: output directory where the results are saved
    @param threshold: threshold to consider a detection ring as false positive. Between 0 and 1.
    @param cx: x coordinate of the pith disk
    @param cy: y coordinate of the pith disk
    @return: Precision, Recall, F-score and mean RMSE
    """
    if threshold > 1:
        raise ValueError("The threshold must be between 0 and 1")

    metrics = InfluenceArea(gt_file, dt_file, img_filename, output_dir, threshold, cx, cy)
    TP, FP, TN, FN = metrics.compute_indicators()

    F = metrics.fscore(TP, FP, TN, FN)
    P = metrics.precision(TP, FP, TN, FN)
    R = metrics.recall(TP, FP, TN, FN)

    RMSE = metrics.compute_rmse_global()
    print(f"{Path(img_filename).name} P={P:.2f} R={R:.2f} F={F:.2f} RMSE={RMSE:.2f}")

    metrics.generate_radial_error_heat_map()

    return P, R, F, RMSE, TP, FP, TN, FN


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dt_filename", type=str, required=True)
    parser.add_argument("--gt_filename", type=str, required=True)
    parser.add_argument("--img_filename", type=str, required=True)
    parser.add_argument("--cx", type=int, required=True, help="x pith coordinate")
    parser.add_argument("--cy", type=int, required=True, help="y pith coordinate")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory for the results")
    parser.add_argument("--th", type=float, required=True,
                        help="threshold to consider a detection as valid. Between 0 and 1")

    args = parser.parse_args()
    main(args.dt_filename, args.gt_filename, args.img_filename, args.output_dir, args.th, args.cx, args.cy)
