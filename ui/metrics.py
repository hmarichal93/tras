import os
import base64
import cv2
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

from typing import List
from streamlit_image_zoom import image_zoom
from pathlib import Path
from shapely.geometry import LineString, MultiLineString, Polygon, Point, MultiPoint

from lib.image import  Color as ColorCV2, Drawing, load_image, write_image
from ui.common import Context
from lib.metrics import  export_results, Table
from backend.labelme_layer import (LabelmeShapeType,
                                   LabelmeObject, LabelmeInterface as UserInterface)


class ViewContext(Context):
    def init_specific_ui_components(self):
        config = self.config["image"]
        self.image_path = self.output_dir / config["image_path"]
        self.units_mode = config["scale"]["unit"]
        self.scale_status = config["scale"]["status"]
        self.pixels_length = config["scale"]["pixels_length"]
        self.know_distance = config["scale"]["know_distance"]
        self.dpi = config["scale"]["dpi"]
        self.tree_planting_date = config["metadata"]["tree_planting_date"]


        config_manual = self.config["manual"]

        self.ew_annotation_file = config_manual["annotations_files"]["early_wood"]
        self.ew_annotation_file = None if len(self.ew_annotation_file) == 0 else self.ew_annotation_file
        self.lw_annotation_file = config_manual["annotations_files"]["late_wood"]
        self.lw_annotation_file = None if len(self.lw_annotation_file) == 0 else self.lw_annotation_file
        self.knot_annotation_file = config_manual["annotations_files"]["knot"]
        self.knot_annotation_file = None if len(self.knot_annotation_file) == 0 else self.knot_annotation_file
        self.cw_annotation_file = config_manual["annotations_files"]["compression_wood"]
        self.cw_annotation_file = None if len(self.cw_annotation_file) == 0 else self.cw_annotation_file

        self.output_dir_metrics = self.output_dir  / "metrics"
        self.output_dir_metrics.mkdir(parents=True, exist_ok=True)

        config_metric = self.config["metric"]
        self.main_label = config_metric["main_label"]
        self.ring_area = config_metric["ring_area"]
        self.ew_area = config_metric["ew_area"]
        self.lw_area = config_metric["lw_area"]
        self.eccentricity_module = config_metric["eccentricity_module"]
        self.perimeter = config_metric["perimeter"]
        self.ew_lw_label = config_metric["ew_lw_label"]
        self.cumulative_area = config_metric["cumulative_area"]
        self.cumulative_ew_area = config_metric["cumulative_ew_area"]

        self.lw_width = config_metric["lw_width"]
        self.eccentricity_phase = config_metric["eccentricity_phase"]
        self.ring_similarity_factor = config_metric["ring_similarity_factor"]
        self.year = config_metric["year"]
        self.cumulative_radius = config_metric["cumulative_radius"]
        self.cumulative_ew_radius = config_metric["cumulative_ew_radius"]
        self.lw_ratio = config_metric["lw_ratio"]
        self.annual_ring_width = config_metric["annual_ring_width"]
        self.ew_width = config_metric["ew_width"]
        self.lw_width_ratio = config_metric["lw_width_ratio"]

        self.ring_path = config_metric["ring_path"]



    def update_config(self):
        config_metric = self.config["metric"]
        config_metric["main_label"] = self.main_label
        config_metric["ring_area"] = self.ring_area
        config_metric["ew_area"] = self.ew_area
        config_metric["lw_area"] = self.lw_area
        config_metric["eccentricity_module"] = self.eccentricity_module
        config_metric["perimeter"] = self.perimeter
        config_metric["ew_lw_label"] = self.ew_lw_label
        config_metric["cumulative_area"] = self.cumulative_area
        config_metric["cumulative_ew_area"] = self.cumulative_ew_area
        config_metric["lw_width"] = self.lw_width
        config_metric["eccentricity_phase"] = self.eccentricity_phase
        config_metric["ring_similarity_factor"] = self.ring_similarity_factor
        config_metric["year"] = self.year
        config_metric["cumulative_radius"] = self.cumulative_radius
        config_metric["cumulative_ew_radius"] = self.cumulative_ew_radius
        config_metric["lw_ratio"] = self.lw_ratio
        config_metric["annual_ring_width"] = self.annual_ring_width
        config_metric["ew_width"] = self.ew_width
        config_metric["lw_width_ratio"] = self.lw_width_ratio
        config_metric["ring_path"] = str(self.ring_path)




def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def select_columns_to_display(CTX, table: Table):
    "select columns to display"
    columns = []
    if CTX.main_label:
        columns.append(table.main_label)
    if CTX.ring_area:
        columns.append(table.ring_area)
    if CTX.ew_area:
        columns.append(table.ew_area)
    if CTX.lw_area:
        columns.append(table.lw_area)
    if CTX.eccentricity_module:
        columns.append(table.eccentricity_module)
    if CTX.perimeter:
        columns.append(table.perimeter)
    if CTX.ew_lw_label:
        columns.append(table.ew_lw_label)
    if CTX.cumulative_area:
        columns.append(table.cumulative_area)
    if CTX.cumulative_ew_area:
        columns.append(table.cumulative_ew_area)
    if CTX.lw_width:
        columns.append(table.lw_width)
    if CTX.eccentricity_phase:
        columns.append(table.eccentricity_phase)
    if CTX.ring_similarity_factor:
        columns.append(table.ring_similarity_factor)
    if CTX.year:
        columns.append(table.year)
    if CTX.cumulative_radius:
        columns.append(table.cumulative_radius)
    if CTX.cumulative_ew_radius:
        columns.append(table.cumulative_ew_radius)
    if CTX.lw_ratio:
        columns.append(table.lw_ratio)
    if CTX.annual_ring_width:
        columns.append(table.annual_ring_width)
    if CTX.ew_width:
        columns.append(table.ew_width)
    if CTX.lw_width_ratio:
        columns.append(table.lw_width_ratio)
    return columns



class UI:

    def __init__(self, runtime_config_path):
        CTX = ViewContext(runtime_config_path)
        CTX.init_specific_ui_components()
        self.CTX = CTX
        self.df = None

    def options(self):
        st.subheader("Columns to display in the table")
        table = Table(self.CTX.units_mode)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            main_label = st.checkbox(table.main_label, self.CTX.main_label)
            self.CTX.main_label = True if main_label else False
            ring_area = st.checkbox(table.ring_area, self.CTX.ring_area)
            self.CTX.ring_area = True if ring_area else False
            ew_area = st.checkbox(table.ew_area, self.CTX.ew_area)
            self.CTX.ew_area = True if ew_area else False
            lw_area = st.checkbox(table.lw_area, self.CTX.lw_area)
            self.CTX.lw_area = True if lw_area else False
            eccentricity_module = st.checkbox(table.eccentricity_module, self.CTX.eccentricity_module)
            self.CTX.eccentricity_module = True if eccentricity_module else False
            perimeter = st.checkbox(table.perimeter, self.CTX.perimeter)
            self.CTX.perimeter = True if perimeter else False



        with col2:
            ew_lw_label = st.checkbox(table.ew_lw_label, self.CTX.ew_lw_label)
            self.CTX.ew_lw_label = True if ew_lw_label else False
            cumulative_area = st.checkbox(table.cumulative_area, self.CTX.cumulative_area)
            self.CTX.cumulative_area = True if cumulative_area else False
            cumulative_ew_area = st.checkbox(table.cumulative_ew_area, self.CTX.cumulative_ew_area)
            self.CTX.cumulative_ew_area = True if cumulative_ew_area else False
            lw_width = st.checkbox(table.lw_width, self.CTX.lw_width)
            self.CTX.lw_width = True if lw_width else False
            eccentricity_phase = st.checkbox(table.eccentricity_phase, self.CTX.eccentricity_phase)
            self.CTX.eccentricity_phase = True if eccentricity_phase else False
            ring_similarity_factor = st.checkbox(table.ring_similarity_factor, self.CTX.ring_similarity_factor)
            self.CTX.ring_similarity_factor = True if ring_similarity_factor else False



        with col3:
            year = st.checkbox(table.year, self.CTX.year)
            self.CTX.year = True if year else False
            cumulative_radius = st.checkbox(table.cumulative_radius, self.CTX.cumulative_radius)
            self.CTX.cumulative_radius = True if cumulative_radius else False
            cumulative_ew_radius = st.checkbox(table.cumulative_ew_radius, self.CTX.cumulative_ew_radius)
            self.CTX.cumulative_ew_radius = True if cumulative_ew_radius else False
            lw_ratio = st.checkbox(table.lw_ratio, self.CTX.lw_ratio)
            self.CTX.lw_ratio = True if lw_ratio else False

        with col4:
            annual_ring_width = st.checkbox(table.annual_ring_width, self.CTX.annual_ring_width)
            self.CTX.annual_ring_width = True if annual_ring_width else False
            ew_width = st.checkbox(table.ew_width, self.CTX.ew_width)
            self.CTX.ew_width = True if ew_width else False
            lw_width_ratio = st.checkbox(table.lw_width_ratio, self.CTX.lw_width_ratio)
            self.CTX.lw_width_ratio = True if lw_width_ratio else False




    def run_metrics(self):
        if not self.CTX.scale_status:
            os.system(f"rm -rf {self.CTX.output_dir_metrics}")
            st.warning("Please set the scale")

        enabled = self.CTX.lw_annotation_file is not None and self.CTX.scale_status
        run_button = st.button("Run", disabled = not enabled)

        if run_button:
            metadata = dict(
                unit = self.CTX.units_mode,
                pixels_millimeter_relation =  self.CTX.know_distance / self.CTX.pixels_length ,
                plantation_date = True,
                year = self.CTX.tree_planting_date['year']

            )
            if Path(self.CTX.lw_annotation_file).exists():
                lw_file_path = self.CTX.output_dir_metrics / "latewood_read.json"
                os.system(f"cp {self.CTX.lw_annotation_file} {lw_file_path}")
                lw_file_path = self.CTX.output_dir_metrics / "latewood.json"
            else:
                lw_file_path = None

            if Path(self.CTX.ew_annotation_file).exists():
                ew_file_path = self.CTX.output_dir_metrics / "earlywood_read.json"
                os.system(f"cp {self.CTX.ew_annotation_file} {ew_file_path}")
                ew_file_path = self.CTX.output_dir_metrics / "earlywood.json"
            else:
                ew_file_path = None

            export_results(labelme_latewood_path= lw_file_path,
                           labelme_earlywood_path= ew_file_path,
                           image_path=self.CTX.image_path,
                           metadata=metadata,
                           draw=True,
                           output_dir=self.CTX.output_dir_metrics)

        self.dataframe_file = self.CTX.output_dir_metrics / "measurements.csv"
        if not Path(self.dataframe_file).exists():
            return None
        #display dataframe_file

        st.write(f"Results are saved in {self.CTX.output_dir_metrics}")

        rings_image_path = self.CTX.output_dir_metrics / "rings.png"
        import cv2
        image = cv2.cvtColor(load_image(rings_image_path), cv2.COLOR_BGR2RGB)



        self.df = pd.read_csv(self.dataframe_file)
        #self.df['Year'] = self.df['Year'].astype(int)

        #select columns to display
        table =  Table(self.CTX.units_mode)
        columns = select_columns_to_display(self.CTX, table)
        self.df = self.df[columns]

        ring_images = [str(image_path) for image_path in self.CTX.output_dir_metrics.glob("*_ring_properties*.png")]
        from natsort import natsorted
        ring_images = natsorted(ring_images)
        base64_images = []
        #get python file path
        script_path = os.path.abspath(__file__)
        root = Path(script_path).parent.parent
        static_files_dir = Path(root) / "static"
        static_files_dir.mkdir(parents=True, exist_ok=True)
        for image_path in ring_images:
            #copy image_path to static_files_dir
            os.system(f"cp {image_path} {static_files_dir}")
            base64_images.append(f"app/static/{Path(image_path).name}")

        self.df['image'] = base64_images
        #put 'image' column as first
        cols = self.df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        self.df = self.df[cols]



        st.data_editor(self.df,
                       column_config= {
                           'image': st.column_config.ImageColumn('Preview Ring', help="Preview Ring")
                       },
                       hide_index = True
        )

        image_zoom(image, mode="scroll", size=(800, 600), keep_aspect_ratio=True, zoom_factor=4.0, increment=0.2)


        #tab1, tab2 = st.tabs(["Line plot", "Bar plot"])

        df_columns = self.df.columns.tolist()
        df_columns.remove('image')
        #df_columns.remove(table.ew_lw_label)
        #df_columns.remove(table.main_label)
        index_year = 0#df_columns.index(table.year)
        index_radius_width = 1#df_columns.index(table.cumulative_radius)
        x_axis = st.selectbox("Select x-axis", df_columns, index=index_year)
        y_axis = st.selectbox("Select y-axis", df_columns, index=index_radius_width)

        if x_axis == y_axis:
            st.error("Please select different columns for x and y axis")
            return

        x_axis_values = self.df[x_axis].values
        y_axis_values = self.df[y_axis].values
        x_axis = x_axis.split("[")[0]
        y_axis = y_axis.split("[")[0]
        df = pd.DataFrame(data={x_axis: x_axis_values, y_axis: y_axis_values}, columns=[x_axis, y_axis])

        chart = alt.Chart(df).mark_line(color="#FF5733").encode(
            x=x_axis,
            y=y_axis
        ).properties(
            width=800,
            height=400,
            title=alt.TitleParams(f"Measurement Unit: {self.CTX.units_mode}", anchor='middle', offset=20)
        )

        st.altair_chart(chart)
        return

    def plot(self, x_axis, y_axis, x_axis_values, y_axis_values):
        x_axis = x_axis.split("[")[0]
        y_axis = y_axis.split("[")[0]
        df = pd.DataFrame(data={x_axis: x_axis_values, y_axis: y_axis_values}, columns=[x_axis, y_axis])

        chart = alt.Chart(df).mark_line(color="#FF5733").encode(
            x=x_axis,
            y=y_axis
        ).properties(
            width=800,
            height=400,
            title=alt.TitleParams(f"Measurement Unit: {self.CTX.units_mode}", anchor='middle', offset=20)
        )

        st.altair_chart(chart)
    def delineate_path(self):
        #add a button for delineating path
        if not self.CTX.scale_status:
            os.system(f"rm -rf {self.CTX.output_dir_metrics}")
            st.warning("Please set the scale")

        enabled = self.CTX.lw_annotation_file is not None and self.CTX.scale_status
        button = st.button("Delineate Path", disabled= not enabled)
        if not enabled:
            st.error("Please upload the latewood annotation file")
            return

        output_path = self.CTX.output_dir_metrics / "coorecorder.csv"

        if button:
            self.CTX.ring_path = self.CTX.output_dir / "path.json"
            interface = PathInterface(self.CTX.image_path, self.CTX.ring_path)
            interface.interface()
            results = interface.parse_output()
            object_lw = LabelmeObject(self.CTX.lw_annotation_file)
            l_intersections = interface.compute_intersections( object_lw, results)

            interface.compute_metrics(l_intersections, output_path,
                                      scale = self.CTX.know_distance / self.CTX.pixels_length,
                                      unit = self.CTX.units_mode)

            st.write(f"Results are saved in {self.CTX.output_dir_metrics}")

        if output_path.exists():
            df = pd.read_csv(output_path)
            columns = df.columns.tolist()
            st.write(df)
            image = load_image(self.CTX.output_dir / "debug_path.png")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_zoom(image, mode="scroll", size=(800, 600), keep_aspect_ratio=True, zoom_factor=4.0, increment=0.2)
            x_axis = "index"
            y_axis = columns[1]
            x_axis_values = np.arange(df.shape[0]) #df[x_axis].values
            y_axis_values = df[y_axis].values
            self.plot(x_axis, y_axis, x_axis_values, y_axis_values)
            y_axis = columns[2]
            y_axis_values = df[y_axis].values
            self.plot(x_axis, y_axis, x_axis_values, y_axis_values)
            st.warning("Index 0 refers to the pith position")


class PathInterface(UserInterface):
    def __init__(self, image_path, output_json_path, output_image_path=None):
        super().__init__(read_file_path = image_path, write_file_path=output_json_path)
        self.output_image_path = output_image_path

    def parse_output(self):
        object = LabelmeObject(self.write_file_path)
        if len(object.shapes) > 1:
            st.error("More than one shape found. Add only one shape")
            return None

        shape = object.shapes[0]
        if not(shape.shape_type == LabelmeShapeType.line or  shape.shape_type == LabelmeShapeType.linestrip):
            st.error("Shape is not a line or linestrip. Remember that you are delineating the ring path")
            return None

        if shape.shape_type == LabelmeShapeType.line:
            return LineString(shape.points)

        if shape.shape_type == LabelmeShapeType.linestrip:
            l_lines = [ LineString(np.concatenate((shape.points[idx].reshape((1,2)), shape.points[idx+1].reshape((1,2)))))
                       for idx in range(shape.points.shape[0]-1)]

            return MultiLineString(l_lines)

        return None


    def compute_intersections(self, rings: LabelmeObject, path: LineString | MultiLineString, debug=True):
        if debug:
            image = load_image(self.read_file_path)
            debug_image_path = self.read_file_path.parent / f"debug_path.png"

        class PointLabelme(Point):
            def __init__(self, x, y, label):
                self.label = label
                super().__init__([ x, y])

        pith = True
        #compute intersections
        l_intersection = []
        for ring in rings.shapes:

            ring_polygon = Polygon(ring.points)
            if pith:
                centroid = ring_polygon.centroid
                pith = False
                l_intersection.append(PointLabelme(x=centroid.x, y=centroid.y, label="Pith"))

            if debug:
                image = Drawing.curve(ring_polygon.exterior, image, color=ColorCV2.red, thickness=2)
            intersection = path.intersects(ring_polygon)
            if intersection:
                #get intersected points
                intersection_points = ring_polygon.exterior.intersection(path)
                #check if intersection_points is empty
                if intersection_points.is_empty:
                    continue

                x, y = intersection_points.coords.xy if not isinstance(intersection_points, MultiPoint) else\
                    intersection_points[0].coords.xy
                l_intersection.append(PointLabelme(x= x[0], y=y[0], label=ring.label))

                if debug:
                    image = Drawing.curve(ring_polygon.exterior, image, color=ColorCV2.red, thickness=2)
                    if isinstance(x, np.ndarray):
                        for idx in range(len(x)):
                            image = Drawing.circle(image, (int(y[idx]), int(x[idx])), radius=5,
                                                   color=ColorCV2.blue, thickness=-1)
                    else:
                        image = Drawing.circle(image, (int(y[0]), int(x[0])), radius=5,
                                               color=ColorCV2.blue, thickness=-1)
                        image = Drawing.put_text(ring.label, image, (int(y[0]), int(x[0])), color=ColorCV2.black, fontScale=1.0)

        if debug:
            write_image(str(debug_image_path), image)

        return l_intersection


    def compute_metrics(self, l_intersection: List, output_path: Path, unit: str, scale: float = 1.0)\
            -> pd.DataFrame:
        from lib.metrics import PathMetrics

        path = PathMetrics(l_intersection, scale, self.read_file_path.name, unit)

        output_path_pos = str(output_path).replace(".csv", ".pos")
        
        path.export_coorecorder_format( output_path = output_path_pos)

        df = path.compute(output_path)

        return df


    def from_structure_to_labelme_shape(self, structure_list):
        pass

    def from_labelme_shape_to_structure(self, shapes):
        pass




def main(runtime_config_path):
    ui = UI(runtime_config_path)

    st.divider()
    tab1, tab2 = st.tabs(["Area-Based", "Path-Based"])

    with tab1:
        st.header("Area-Based Metrics")
        ui.options()

        st.divider()
        selected = ui.run_metrics()

    with tab2:
        st.header("Path-Based Metrics")
        ui.delineate_path()

    st.divider()


    ui.CTX.save_config()
    return

def annotate_pith():
    #TODO: Implement
    pass
