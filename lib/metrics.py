"""
Module for computing important metrics
"""
import numpy as np
import pandas as pd

from shapely.geometry import Polygon, LineString

import numpy as np
import cv2
import datetime

from pathlib import Path

import pandas as pd
from shapely.geometry import Polygon, Point

from lib.image import Color, Drawing

from backend.labelme_layer import AL_AnnualRings


def export_results( labelme_latewood_path : str, labelme_earlywood_path : str, image_path : str, metadata: dict,
                   output_path="output/measures.csv", draw=True, plantation_date=True):
    #metadata
    year = metadata["year"]
    year = datetime.datetime(year, 1, 1)

    pixels_millimeter_relation = float(metadata["pixels_millimeter_relation"])

    image = cv2.imread(image_path)
    al_annual_rings = AL_AnnualRings(late_wood_path=Path(labelme_latewood_path),
                                     early_wood_path=Path(labelme_earlywood_path))
    annual_rings_list = al_annual_rings.read()

    df = pd.DataFrame(columns=["Annual Ring (main_label)", "Annual Ring (secondary label)", "Year",
                               "Area [mm2]", "Area EW [mm2]", "Area LW [mm2]", "Area LW/EW (%)",
                               "Width Annual Ring [mm]", "Width EW [mm]", "Width LW [mm]", "Width LW/EW (%)",
                               "Eccentricity Module [mm]", "Eccentricity Phase [°]", "Ring Similarity Factor [0-1]"])

    pith = Point(0, 0)
    image_full = image.copy()
    for idx, ring in enumerate(annual_rings_list):
        #area
        area = ring.area
        latewood_area = ring.late_wood.area if ring.late_wood is not None else 0
        earlywood_area = ring.early_wood.area if ring.early_wood is not None else 0
        area_latewood_earlywood = None #latewood_area / earlywood_area if latewood_area > 0 and earlywood_area > 0 else 0

        #width
        width_annual_ring = None #ring.equivalent_radii()
        width_latewood = None #ring.late_wood.equivalent_radii() if ring.late_wood is not None else 0
        width_earlywood = None #ring.early_wood.equivalent_radii() if ring.early_wood is not None else 0
        width_latewood_earlywood = None #width_latewood / width_earlywood if width_latewood > 0 and width_earlywood > 0 else 0

        #eccentricity
        if idx == 0:
            pith = ring.centroid
        eccentricity_module = ring.centroid.distance(pith)
        if eccentricity_module == 0:
            eccentricity_phase = 0
        else:
            x, y = (ring.centroid - pith).coords.xy
            eccentricity_phase = np.arctan2(y, x)[0]* 180 / np.pi if x != 0 else 0

        ring_similarity_factor = ring.similarity_factor()

        #save results
        df.loc[idx] = [
            f"{ring.main_label}", f"{ring.secondary_label}", year.year,
            area * (pixels_millimeter_relation**2), earlywood_area * (pixels_millimeter_relation**2),
            latewood_area * (pixels_millimeter_relation**2) , area_latewood_earlywood * (pixels_millimeter_relation),
            width_annual_ring * pixels_millimeter_relation, width_earlywood * pixels_millimeter_relation ,
            width_latewood * pixels_millimeter_relation , width_latewood_earlywood * pixels_millimeter_relation,
            eccentricity_module * pixels_millimeter_relation, eccentricity_phase, ring_similarity_factor
        ]
        image_full = ring.draw_rings(image_full, thickness=3)
        #increase year by 1 year more
        year = year + datetime.timedelta(days=365) if plantation_date else year - datetime.timedelta(days=365)
        if draw:
            thickness= 3
            image_debug = ring.draw(image.copy(), full_details=True, opacity=0.1)
            image_debug = Drawing.curve(ring.exterior.coords, image_debug, Color.black, thickness)
            inner_points = np.array([list(interior.coords) for interior in ring.interiors]).squeeze()
            if len(inner_points) > 0:
                aux_poly = Polygon(inner_points)
                image_debug = Drawing.curve(aux_poly.exterior.coords, image_debug, Color.black, thickness)
            output_name = f"output/{ring.main_label}.png"
            cv2.imwrite(output_name, image_debug)

    cv2.imwrite("output/rings.png", image_full)
    #format df %.02f
    df = df.round(2)
    df.to_csv(output_path, index=False)
    return



def main():
    root = "./input/C14/"
    image_path = f"{root}image.jpg"
    labelme_latewood_path = f"{root}latewood.json"
    labelme_earlywood_path = f"{root}earlywood.json"
    metadata = {
        "year": 1993,
        "pixels_millimeter_relation": 10 / 52
    }
    export_results(labelme_latewood_path, labelme_earlywood_path, image_path, metadata)




if __name__ == "__main__":
    main()