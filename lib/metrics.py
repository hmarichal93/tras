"""
Module for computing important metrics
"""
import numpy as np
import cv2
import pandas as pd

from pathlib import Path
from shapely.geometry import Polygon, Point, LineString
from typing import List
import datetime

from lib.image import Color, Drawing, resize_image_using_pil_lib, load_image, write_image
from backend.labelme_layer import AL_AnnualRings, AL_LateWood_EarlyWood


class Table:
    def __init__(self, unit="mm"):
        self.unit = unit
        self.main_label = "Annual Ring (label)"
        self.ew_lw_label = "EW/LW label"
        self.year = "Year"

        self.ring_area = f"Ring Area [{unit}2]"
        self.cumulative_area = f"Cumulative Area [{unit}2]"
        self.cumulative_radius = f"Cumulative Annual Radius [{unit}]"
        self.annual_ring_width = f"Annual Ring Width [{unit}]"

        self.ew_area = f"Area EW [{unit}2]"
        self.cumulative_ew_area = f"Cumulative R(n-1) + EW(n) Area [{unit}2]"
        self.cumulative_ew_radius = f"Cumulative EW Radius [{unit}]"
        self.ew_width = f"EW Width [{unit}]"

        self.lw_area = f"Area LW [{unit}2]"
        self.lw_width = f"LW Width [{unit}]"
        self.lw_ratio = f"Area LW/(LW +EW) (-)"
        self.lw_width_ratio = f"Width LW/(LW +EW) (-)"

        self.eccentricity_module = f"Eccentricity Module [{unit}]"
        self.eccentricity_phase = f"Eccentricity Phase [°]"
        self.perimeter = f"Perimeter [{unit}]"
        self.ring_similarity_factor = f"Ring Similarity Factor [0-1]"

        self.exclusion_area = "Exclusion Area"


def fill_df(annual_ring_label_list, year_list, ew_lw_label_list, ring_area_list, ew_area_list, eccentricity_module_list,
            eccentricity_phase_list, ring_perimeter_list, pixels_millimeter_relation, unit):
    table = Table(unit=unit)
    df = pd.DataFrame(columns=[table.main_label, table.ew_lw_label, table.year,  # metadata
                               table.ring_area,  # ring properties
                               table.cumulative_area, table.cumulative_radius, table.annual_ring_width,
                               # math operations on ring properties
                               table.ew_area,  # ring properties
                               table.cumulative_ew_area, table.cumulative_ew_radius, table.ew_width,
                               # math operations on ring properties
                               table.lw_area,  # ring properties
                               table.lw_width,  # math operations on ring properties
                               table.lw_ratio, table.lw_width_ratio,  # math operations on ring properties
                               table.eccentricity_module, table.eccentricity_phase,  # ring properties
                               table.perimeter,  # ring properties
                               table.ring_similarity_factor]  # math operations on ring properties
                      )

    df[table.main_label] = annual_ring_label_list
    df[table.ew_lw_label] = ew_lw_label_list
    df[table.year] = year_list

    df[table.ring_area] = np.array(ring_area_list) * (pixels_millimeter_relation ** 2)
    df[table.cumulative_area] = df[table.ring_area].cumsum()
    df[table.cumulative_radius] = np.sqrt(df[table.cumulative_area] / np.pi)
    annual_ring_width_list = df[table.cumulative_radius].diff()
    annual_ring_width_list[0] = df[table.cumulative_radius].iloc[0]
    df[table.annual_ring_width] = np.array(annual_ring_width_list)

    df[table.ew_area] = np.array(ew_area_list) * (pixels_millimeter_relation ** 2)
    df[table.cumulative_ew_area] = (df[table.cumulative_area].shift(1) + df[table.ew_area]).fillna(0)
    df[table.cumulative_ew_radius] = np.sqrt(df[table.cumulative_ew_area] / np.pi)
    ew_ring_width_list = (df[table.cumulative_ew_radius] - df[table.cumulative_radius].shift(1)).fillna(0)
    df[table.ew_width] = np.array(ew_ring_width_list)

    df[table.lw_area] = df[table.ring_area] - df[table.ew_area]
    df[table.lw_width] = df[table.annual_ring_width] - df[table.ew_width]
    df[table.lw_ratio] = df[table.lw_area] / df[table.ring_area]
    df[table.lw_width_ratio] = df[table.lw_width] / df[table.annual_ring_width]
    df[table.eccentricity_module] = np.array(eccentricity_module_list) * pixels_millimeter_relation
    df[table.eccentricity_phase] = np.array(eccentricity_phase_list)
    df[table.perimeter] = np.array(ring_perimeter_list) * pixels_millimeter_relation
    df[table.ring_similarity_factor] = 1 - (df[table.perimeter] - 2 * np.pi * df[table.cumulative_radius]) / \
                                       df[table.perimeter]
    df = df.round(2)
    return df, table


def compute_angle(vector):
    x, y = vector
    angle_radians = np.arctan2(y, x)
    angle_degrees = np.degrees(angle_radians)
    angle_360 = angle_degrees if angle_degrees >= 0 else angle_degrees + 360
    return angle_360


def extract_ring_properties(annual_rings_list, year, plantation_date=False, exclusion_shapes: List = None):
    pith = Point(0, 0)

    ring_area_list = []
    ew_area_list = []
    lw_area_list = []
    ring_perimeter_list = []
    eccentricity_module_list = []
    eccentricity_phase_list = []
    year_list = []
    annual_ring_label_list = []
    ew_lw_label_list = []
    year = year - (len(annual_rings_list) -1)* datetime.timedelta(days=365)
    for idx, ring in enumerate(annual_rings_list):
        #area
        if exclusion_shapes is None:
            ring_area = ring.area
            latewood_area = ring.late_wood.area if ring.late_wood is not None else 0
            earlywood_area = ring.early_wood.area if ring.early_wood is not None else 0

        else:
            ring_area = ring.compute_non_intersecting_area(exclusion_shapes)
            latewood_area = ring.late_wood.compute_non_intersecting_area(exclusion_shapes) \
                if ring.late_wood is not None else 0
            earlywood_area = ring.early_wood.compute_non_intersecting_area(exclusion_shapes) \
                if ring.early_wood is not None else 0

        ring_area_list.append(ring_area)
        lw_area_list.append(latewood_area)
        ew_area_list.append(earlywood_area)

        #eccentricity
        if idx == 0:
            pith = ring.centroid
        ring_centroid = ring.get_centroid()
        eccentricity_module = ring_centroid.distance(pith)
        if eccentricity_module == 0:
            eccentricity_phase = 0
        else:
            #change reference y-axis to the opposite direction
            convert_to_numpy = lambda point: np.multiply(np.array([point.coords.xy[0], point.coords.xy[1]]).squeeze(),
                                                         np.array([-1, 1]))
            numpy_ring_centroid = convert_to_numpy(ring_centroid).squeeze()

            numpy_pith = convert_to_numpy(pith).squeeze()
            #change origin to pith
            numpy_ring_centroid_referenced_to_pith = numpy_ring_centroid - numpy_pith
            #normalize
            numpy_ring_centroid_referenced_to_pith_normalized = numpy_ring_centroid_referenced_to_pith / np.linalg.norm(
                numpy_ring_centroid_referenced_to_pith)

            angle = compute_angle(numpy_ring_centroid_referenced_to_pith_normalized)
            eccentricity_phase = angle

        eccentricity_module_list.append(eccentricity_module)
        eccentricity_phase_list.append(eccentricity_phase)
        ring_perimeter_list.append(ring.exterior.length)

        #metadata
        year_list.append(year.year)
        annual_ring_label_list.append(ring.main_label)
        ew_lw_label_list.append(ring.secondary_label)
        #save results
        year = year + datetime.timedelta(days=366) #if not plantation_date else year - datetime.timedelta(days=365)

    return annual_ring_label_list, year_list, ew_lw_label_list, ring_area_list, ew_area_list, eccentricity_module_list, eccentricity_phase_list, ring_perimeter_list


def debug_images(annual_rings_list, df, image_path, output_dir):
    if Path(output_dir).exists():
        import os
        os.system(f"rm -rf {output_dir}/*ring_properties_label*")

    image = load_image(image_path)
    image_full = image.copy()
    for idx, ring in enumerate(annual_rings_list):
        #eccentricity
        if idx == 0:
            pith = ring.centroid
        ring_centroid = ring.get_centroid()
        image_full = ring.draw_rings(image_full, thickness=3)
        thickness = 3
        image_debug = ring.draw(image.copy(), full_details=True, opacity=0.1)
        image_debug = Drawing.curve(ring.exterior.coords, image_debug, Color.black, thickness)
        inner_points = np.array([list(interior.coords) for interior in ring.interiors]).squeeze()
        if len(inner_points) > 0:
            aux_poly = Polygon(inner_points)
            image_debug = Drawing.curve(aux_poly.exterior.coords, image_debug, Color.black, thickness)
            #draw arrow from centroid to pith
        image_debug = Drawing.arrow(image_debug, pith, ring_centroid, Color.red, thickness=3)
        output_name = f"{output_dir}/{idx}_ring_properties_label_{ring.main_label}.png"
        image_debug = resize_image_using_pil_lib(image_debug, 640, 640)
        write_image(output_name, image_debug)

    #image_full = resize_image_using_pil_lib(image_full, 640, 640)
    write_image(f"{output_dir}/rings.png", image_full)

    return


def load_exclusion_shapes(exclusion_shape_path: str = None):
    if exclusion_shape_path is None:
        return None

    al_exclusion = AL_LateWood_EarlyWood(exclusion_shape_path, None)
    exclusion_shapes = al_exclusion.read()
    exclusion_shapes = [Polygon(shape.points) for shape in exclusion_shapes]
    return exclusion_shapes

def compute_area_based_properties(labelme_latewood_path: str = None, labelme_earlywood_path: str = None,
                                  image_path: str = None,
                                  metadata: dict = None,
                                  output_dir="output", draw=False, code=None,
                                  exclusion_shape_path: str = None):
    #metadata
    year = metadata["year"]
    year = datetime.datetime(year, 1, 1)

    plantation_date = metadata["plantation_date"]

    pixels_millimeter_relation = float(metadata["pixels_millimeter_relation"])

    unit = metadata.get("unit", "mm")

    exclusion_shapes = load_exclusion_shapes(exclusion_shape_path)

    al_annual_rings = AL_AnnualRings(late_wood_path=Path(labelme_latewood_path),
                                     early_wood_path=Path(labelme_earlywood_path) if labelme_earlywood_path else None)
    annual_rings_list = al_annual_rings.read()

    (annual_ring_label_list, year_list, ew_lw_label_list, ring_area_list, ew_area_list, eccentricity_module_list,
     eccentricity_phase_list, ring_perimeter_list) = extract_ring_properties(annual_rings_list, year,
                                                                             exclusion_shapes=exclusion_shapes)

    df, table = fill_df(
        annual_ring_label_list, year_list, ew_lw_label_list, ring_area_list, ew_area_list,
        eccentricity_module_list, eccentricity_phase_list, ring_perimeter_list, pixels_millimeter_relation, unit
    )

    df.to_csv(f"{output_dir}/{code}_measurements.csv", index=False)
    if draw:
        debug_images(annual_rings_list, df, image_path, output_dir)

    generate_plots(table, df, output_dir)
    generate_pdf(df, output_dir)
    return


def generate_plots(table, df, output_dir):
    #pass
    #Area bar plot
    lw_area = df[table.lw_area]
    ew_area = df[table.ew_area]
    ring_area = df[table.ring_area]
    year = df["Year"]
    #convert year to int
    year = year.astype(int)
    from matplotlib import pyplot as plt
    plt.figure()
    bar_width = 0.25
    plt.bar(year - bar_width / 2.1, ew_area, label="Earlywood", width=bar_width)
    plt.bar(year - bar_width / 2.1, lw_area, bottom=ew_area, label="Latewood", width=bar_width)
    plt.bar(year + bar_width / 2.1, ring_area, label="Ring", width=bar_width)

    plt.xticks(year)
    plt.grid(True)
    #rotate xticks 90 degrees
    plt.xticks(rotation=90)
    plt.xlabel("Year")
    plt.ylabel(f"Area [{table.unit}2]")
    plt.legend()
    plt.title("Ring Area Distribution")
    plt.savefig(f"{output_dir}/area_bar_plot.png")
    plt.close()

    #ring width bar plot
    lw_width = df[table.lw_width]
    ew_width = df[table.ew_width]
    ring_width = df[table.annual_ring_width]
    plt.figure()
    plt.bar(year - bar_width / 2.1, ew_width, label="Earlywood", width=bar_width)
    plt.bar(year - bar_width / 2.1, lw_width, bottom=ew_width, label="Latewood", width=bar_width)
    plt.bar(year + bar_width / 2.1, ring_width, label="Ring", width=bar_width)

    plt.xticks(year)
    plt.grid(True)
    #rotate xticks 90 degrees
    plt.xticks(rotation=90)
    plt.xlabel("Year")
    plt.ylabel(f"Width [{table.unit}]")
    plt.legend()
    plt.title("Ring Width Distribution")
    plt.savefig(f"{output_dir}/width_bar_plot.png")
    plt.close()

    #ring cummulatives plot
    ring_width = df[table.cumulative_radius]
    plt.figure()
    plt.plot(year, ring_width)
    plt.xticks(year)
    plt.grid(True)
    #rotate xticks 90 degrees
    plt.xticks(rotation=90)
    plt.xlabel("Year")
    plt.ylabel(f"Radius [{table.unit}]")
    plt.title("Ring Cumulative Radius")
    plt.savefig(f"{output_dir}/radius_plot.png")
    plt.close()
    return


def generate_pdf(df, output_dir):
    #generate pdf with plots
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.image(f"{output_dir}/rings.png", x=10, y=30, w=180)
    pdf.add_page()
    # pdf.set_font("Arial", size=12)
    # pdf.cell(200, 10, txt="Annual Ring Metrics", ln=True, align="C")
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Ring Area Distribution", ln=True, align="L")
    pdf.image(f"{output_dir}/area_bar_plot.png", x=10, y=30, w=180)
    pdf.add_page()

    pdf.image(f"{output_dir}/width_bar_plot.png", x=10, y=30, w=180)
    pdf.add_page()

    pdf.image(f"{output_dir}/radius_plot.png", x=10, y=30, w=180)
    pdf.add_page()

    #extra pages for more details
    #add page title
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Ring Details", ln=True, align="C")

    images_list = [f"{output_dir}/{idx}_ring_properties_label_{df.iloc[idx]["Annual Ring (label)"]}.png" for idx in
                   range(df.shape[0])]
    for idx, image in enumerate(images_list):
        #add title "Ring idx"
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt=f"Ring {df.iloc[idx]['Annual Ring (label)']}", ln=True, align="L")
        pdf.image(image, x=10, y=30, w=180)
        pdf.add_page()

    pdf.output(f"{output_dir}/metrics.pdf")


class PathMetrics:
    def __init__(self, l_points: List, scale: float, image_name:str, unit: str):
        self.l_points = l_points
        self.scale = scale
        self.image_name = image_name
        self.unit = unit
        class Columns:
            x = "x [px]"
            y = "y [px]"
            label = "label"
            width = f"Width [{self.unit}]"
            cumulative = f"Cumulative Width [{self.unit}]"

        self.Columns = Columns

    def export_coorecorder_format(self, dpi: int = 2400, output_path: Path = None,
                                  image_name: str = None, pixel_per_mm : float = 1) -> None:

        exporter = CooRecorderExporter()
        exporter.write(output_file=output_path, image_name=image_name,
                       measure_points=self.l_points, pixel_per_mm=pixel_per_mm)

        return

    def compute(self) -> pd.DataFrame:


        df = pd.DataFrame( data = {
                self.Columns.label: [point.label for point in self.l_points],
                self.Columns.x: [point.x for point in self.l_points],
                self.Columns.y: [point.y for point in self.l_points]
            }
        )

        df[self.Columns.width] = self._compute_ring_width(df)
        df[self.Columns.width] = df[self.Columns.width].fillna(0)

        # commulative width
        df[self.Columns.cumulative] = df[self.Columns.width].cumsum()
        df.round(3)

        return df[[self.Columns.label, self.Columns.width, self.Columns.cumulative, self.Columns.x, self.Columns.y]]

    def _compute_ring_width(self, df):
        x = df[self.Columns.x].values
        x_shift = df[self.Columns.x].shift(1).values

        y = df[self.Columns.y].values
        y_shift = df[self.Columns.y].shift(1).values
        width = np.sqrt(((x - x_shift) ** 2 + (y - y_shift) ** 2)) * self.scale

        return width

class CooRecorderExporter:
    def __init__(self):
        pass

    def read(self):
        pass

    def write(self, output_file : str = None, image_name : str = None,
              measure_points: List[Point] = None, pixel_per_mm : float = 1, coordinates_in_pixels = True) -> None:
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
        # Prepare unit conversion
        dpi = float(25.4 * pixel_per_mm) if not coordinates_in_pixels else 25.4
        # Create paths for output files

        str_measure_points = self.convert_measure_points_to_str(measure_points, pixel_per_mm, coordinates_in_pixels)

        # Write to file
        coordinates_str = "All coordinates in pixels" if coordinates_in_pixels else "All coordinates in millimeters (mm)"
        with open(output_file, 'w') as file:
            file.write(f"#DENDRO (Cybis Dendro program compatible format) Coordinate file written as\n"
                       f"#Imagefile {image_name}\n#DPI {dpi}\n#{coordinates_str}\n"
                       f"SCALE 1\n#C DATED\n#C Written={dt_string};\n#C CooRecorder=;\n#C licensedTo=;\n")
            for point in str_measure_points:
                file.write(f"{point}\n")



    def convert_measure_points_to_str(self, measure_points: List[Point], pixel_per_mm: float, coordinates_in_pixels = True) \
            -> List[str]:
        number_to_string = lambda x: str(round( x / pixel_per_mm, 5))
        if coordinates_in_pixels:
            return [f"{int(point.y)},{int(point.x)}" for point in measure_points]
        else:
            return [f"{number_to_string(int(point.y))},{number_to_string(int(point.x))}" for point in measure_points]


def main():
    folder_name = "C14"
    #folder_name = "W_F09_T_S2"
    #folder_name = "W_F12_T_S4"
    root = f"./input/{folder_name}/"
    #root = "./output/"
    image_path = f"{root}image.png"
    labelme_latewood_path = f"{root}latewood.json"
    labelme_earlywood_path = f"{root}earlywood.json"
    metadata = {
        "year": 2007,
        "plantation_date": True,
        "pixels_millimeter_relation": 1 / 2.26,  #10 / 52,
        "unit": "micrometer"
    }
    output_dir = f"./output/{folder_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    compute_area_based_properties(labelme_latewood_path, labelme_earlywood_path, image_path, metadata, draw=True,
                                  output_dir=output_dir)
    #export_results(labelme_latewood_path = labelme_latewood_path, image_path= image_path, metadata=metadata, draw=True, output_dir=output_dir)



if __name__ == "__main__":
    main()
