import os
import base64
import streamlit as st
import numpy as np
import pandas as pd

from typing import List
from streamlit_image_zoom import image_zoom
from pathlib import Path
from shapely.geometry import LineString, MultiLineString, Polygon, Point, MultiPoint

from lib.image import  Color as ColorCV2, Drawing, load_image, write_image, resize_image_using_pil_lib
from ui.common import (Context, RunningWidget,  plot_chart, display_image_with_zoom, display_data_editor, check_image,
                       Shapes, display_image_plotly)
from lib.metrics import  export_results, Table
from backend.labelme_layer import (LabelmeShapeType,
                                   LabelmeObject, LabelmeInterface as UserInterface, add_prefix_to_labels,
                                   AL_LateWood_EarlyWood, load_ring_shapes, write_ring_shapes, PointLabelme)


class MultiLineStringLabel(MultiLineString):
    def __init__(self, lines, label):
        super().__init__(lines)
        self.label = label

class LineStringLabel(LineString):
    def __init__(self, points, label):
        super().__init__(points)
        self.label = label

class ViewContext(Context):
    def init_specific_ui_components(self):
        config = self.config["image"]
        self.image_path = self.output_dir / config["image_path"]
        self.units_mode = config["scale"]["unit"]
        self.scale_status = config["scale"]["status"]
        self.pixels_length = config["scale"]["pixels_length"]
        self.know_distance = config["scale"]["know_distance"]
        self.pixel_per_mm = config["scale"]["pixel_per_mm"]
        self.dpi = config["scale"]["dpi"]
        self.harvest_date = config["metadata"]["harvest_date"]
        self.code = config['metadata']['code']


        config_manual = self.config["manual"]

        self.ew_annotation_file = config_manual["annotations_files"]["early_wood"]
        self.ew_annotation_file = None if len(self.ew_annotation_file) == 0 else self.ew_annotation_file
        self.lw_annotation_file = config_manual["annotations_files"]["late_wood"]
        self.lw_annotation_file = None if len(self.lw_annotation_file) == 0 else self.lw_annotation_file


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
        self.ew_measurements = config_metric["ew_measurements"]
        self.two_dim_annotations = config_metric["two_dim_annotations"]

        self.ring_path = config_metric["ring_path"]
        self.display_area_settings = config_metric["display_area_settings"]



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
        config_metric["ew_measurements"] = self.ew_measurements
        config_metric["two_dim_annotations"] = self.two_dim_annotations
        config_metric["display_area_settings"] = self.display_area_settings




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
        st.header("Metrics")
        st.markdown(
            """
            This interface allows you to edit the ring annotations. You can select the shape you want to edit.
            """
        )
        st.divider()
        CTX = ViewContext(runtime_config_path)
        CTX.init_specific_ui_components()
        self.CTX = CTX
        self.df = None

    def options(self):
        display_settings = st.checkbox("Show Area Metrics", value = self.CTX.display_area_settings,
                           help = "Show area metrics to be computed")
        if display_settings != self.CTX.display_area_settings:
            self.CTX.display_area_settings = display_settings
        return
    def show_area_settings(self):
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



    def area_computations(self):
        metadata = dict(
            unit = self.CTX.units_mode,
            pixels_millimeter_relation = self.CTX.know_distance / self.CTX.pixels_length ,
            plantation_date = True,
            year = self.CTX.harvest_date['year']
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
                       output_dir=self.CTX.output_dir_metrics, code=self.CTX.code)

        return

    def area_dataframe_logic(self):
        self.df = pd.read_csv(self.dataframe_file)
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
        if static_files_dir.exists():
            os.system(f"rm -rf {static_files_dir}")

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
        return self.df

    def area_chart(self):
        df_columns = self.df.columns.tolist()
        df_columns.remove('image')
        index_year = 0#df_columns.index(table.year)
        index_radius_width = 1#df_columns.index(table.cumulative_radius)
        x_axis = st.selectbox("Select x-axis", df_columns, index=index_year)
        y_axis = st.selectbox("Select y-axis", df_columns, index=index_radius_width)

        if x_axis == y_axis:
            st.error("Please select different columns for x and y axis")
            return

        x_axis_values = self.df[x_axis].values
        y_axis_values = self.df[y_axis].values
        self.plot(x_axis, y_axis, x_axis_values, y_axis_values)
        return

    def area_metrics(self):
        if self.check_scale():
            return

        if self.check_lw_file():
            return
        #self.options()
        col1, col2, col3 = st.columns([0.3, 0.3, 1])
        with col2:
            display_settings = st.checkbox("Show Area Metrics", value = self.CTX.display_area_settings,
                               help = "Show area metrics to be computed")
            if display_settings != self.CTX.display_area_settings:
                self.CTX.display_area_settings = display_settings

        with col1:
            run_button = st.button("Run")


        if display_settings:
            st.divider()
            self.show_area_settings()

        if run_button:
            gif_running = RunningWidget()
            self.area_computations()
            gif_running.empty()

            self.dataframe_file = self.CTX.output_dir_metrics / f"{self.CTX.code}_measurements.csv"
            if not Path(self.dataframe_file).exists():
                return None


            st.write(f"Results are saved in {self.CTX.output_dir_metrics}")

            tab1, tab2, tab3 = st.tabs(["Image", "table", "Charts"])
            with tab2:
                self.df = self.area_dataframe_logic()
                display_data_editor(self.df)

            with tab1:
                rings_image_path = self.CTX.output_dir_metrics / "rings.png"
                display_image_plotly(rings_image_path)

            with tab3:
                self.area_chart()

        return

    def plot(self, x_axis, y_axis, x_axis_values, y_axis_values):
        x_axis = x_axis.split("[")[0]
        y_axis = y_axis.split("[")[0]
        df = pd.DataFrame(data={x_axis: x_axis_values, y_axis: y_axis_values}, columns=[x_axis, y_axis])
        plot_chart(df, title=f"Measurement Unit: {self.CTX.units_mode}")


    def check_ew_measurements(self):
        ew_measurements = st.checkbox("EW", value = self.CTX.ew_measurements, help = "Add early wood measurements")
        if ew_measurements != self.CTX.ew_measurements:
            self.CTX.ew_measurements = ew_measurements

    def check_scale(self):
        if not self.CTX.scale_status:
            os.system(f"rm -rf {self.CTX.output_dir_metrics}")
            st.error("Please set the scale in the *Image* page")
            return True

        return False

    def check_if_exist_ew_file(self):
        if self.CTX.ew_measurements:
            if not Path(self.CTX.ew_annotation_file).exists():
                st.error("Please upload the earlywood annotation file in the *Ring Editing* page")
                return True
        return False

    def check_lw_file(self):
        enabled = self.CTX.lw_annotation_file is not None and self.CTX.scale_status
        if not enabled:
            st.error("Please upload the latewood annotation file in the *Ring Editing* page")
            return True
        return False

    @staticmethod
    def ring_shapes_processing(json_path, image_path, prefix, flag):
        output_path = Path(json_path).parent / f"tmp.json"
        add_prefix_to_labels(json_path, image_path, prefix, output_path)
        shapes = load_ring_shapes(output_path)
        for s in shapes:
            s.set_flag({'0': flag})

        return shapes

    def compute_intersection(self, interface, path, use_annotations=True):
        if not use_annotations:
            l_intersections = []
            year = int(self.CTX.harvest_date['year']) - (len(path) -1)
            for idx, l in enumerate(path):
                x, y = l.coords[0]
                l_intersections.append(PointLabelme(x=x, y=y, label=f"{year+ idx}"))

            return l_intersections

        if self.CTX.ew_measurements:
            shapes_ew = self.ring_shapes_processing(self.CTX.ew_annotation_file, self.CTX.image_path,
                                                    "ew",Shapes.earlywood)
            shapes_lw = self.ring_shapes_processing(self.CTX.lw_annotation_file, self.CTX.image_path, "lw",
                                                    Shapes.latewood)

            shapes = shapes_lw + shapes_ew
            shapes.sort(key=lambda x: x.area)
            #write shapes to labelme json
            output_path_ann = self.CTX.output_dir_metrics / "lw_ew.json"
            write_ring_shapes(shapes, output_path_ann, self.CTX.image_path)

        else:
            output_path_ann = self.CTX.lw_annotation_file

        object = LabelmeObject(output_path_ann)
        l_intersections = interface.compute_intersections(object, path)
        return l_intersections

    def path_computation(self, output_path):
        self.CTX.ring_path = self.CTX.output_dir / "path.json"
        if self.CTX.ring_path.exists():
            os.system(f"rm {self.CTX.ring_path}")

        interface = PathInterface(self.CTX.image_path, self.CTX.ring_path)
        interface.interface()
        if not self.CTX.ring_path.exists():
            st.error("Path not delineated or saved. Please export the path")
            return []

        path = interface.parse_output(two_dim_annotations=self.CTX.two_dim_annotations)
        if path is None:
            return []

        l_intersections = self.compute_intersection(interface, path, use_annotations= self.CTX.two_dim_annotations)


        self.CTX.path_df_file = Path(f"{output_path}_{path.label}.csv")
        self.CTX.path_coorecorder_file = Path(f"{output_path}_{path.label}.pos")
        self.CTX.path_image_file = Path(f"{output_path}_{path.label}.jpg")
        interface.compute_metrics(l_intersections, coorecorder_output_path = self.CTX.path_coorecorder_file,
                                  coorecorder_image_name = self.CTX.path_image_file.name,
                                  csv_output_path = self.CTX.path_df_file,
                                  unit=self.CTX.units_mode,
                                  pixel_per_mm = self.CTX.pixel_per_mm)
        self.CTX.path_label = path.label

        return l_intersections
    def path_display_results(self, output_path, points):
        st.divider()
        st.markdown("To avoid confusion when earlywood and latewood measurements are calculated together, "
                    "earlywood measurements are prefixed with 'ew,' and latewood measurements are prefixed"
                    " with 'lw.'")
        tab1, tab2, tab3 = st.tabs(["Image", "table", "Charts"])
        with tab2:
            df = pd.read_csv(output_path)
            #set two decimal
            df = df.round(2)
            columns = df.columns.tolist()
            st.write(df)

        with tab1:
            image_path = self.CTX.output_dir / "debug_path.png"
            display_image_plotly(image_path, points, df , save_image=True, output_path = self.CTX.path_image_file)

        with tab3:
            x_axis = "year"
            y_axis = columns[2]
            x_axis_values = df.label.values[1:]
            y_axis_values = df[y_axis].values[1:]
            self.plot(x_axis, y_axis, x_axis_values, y_axis_values)
            y_axis = columns[3]
            y_axis_values = df[y_axis].values[1:]
            self.plot(x_axis, y_axis, x_axis_values, y_axis_values)

        return

    def check_2d_annotations(self):
        check = st.checkbox("2D Annotations", value = self.CTX.two_dim_annotations,
                            help = "Use two 2D annotations. Otherwise, ring boundaries need to be marked manually using the LineStrip object."
                                   " Measurements start from the first marked point, but the last marked point "
                                   "is not included.")
        if check != self.CTX.two_dim_annotations:
            self.CTX.two_dim_annotations = check


    def path_metrics(self):
        if self.check_scale():
            return

        if self.check_lw_file():
            return

        col1, col2, col3 = st.columns([0.3,0.3, 1])

        with col3:
            self.check_2d_annotations()

        with col2:
            self.check_ew_measurements()


        with col1:
            button = st.button("Delineate Path")

        output_path_metrics = self.CTX.output_dir_metrics / "path"
        output_path_metrics.mkdir(parents=True, exist_ok=True)
        output_path = output_path_metrics / self.CTX.code

        if button:
            if self.check_if_exist_ew_file():
                return

            gif_running = RunningWidget()
            res = self.path_computation(output_path)
            gif_running.empty()
            if len(res)==0 :
                return

            st.write(f"Results are saved in {output_path_metrics}")
            self.path_display_results(self.CTX.path_df_file, res)




########################################################################################################################
################################INTERFACE CLASSES#######################################################################
########################################################################################################################


class PathInterface(UserInterface):
    def __init__(self, image_path, output_json_path, output_image_path=None):
        super().__init__(read_file_path = image_path, write_file_path=output_json_path)
        self.output_image_path = output_image_path

    def parse_output(self, two_dim_annotations = True):
        object = LabelmeObject(self.write_file_path)
        if len(object.shapes) > 1:
            st.error("More than one shape found. Add only one shape")
            return None

        shape = object.shapes[0]


        if not(shape.shape_type == LabelmeShapeType.line or  shape.shape_type == LabelmeShapeType.linestrip):
            st.error("Shape is not a line or linestrip. Remember that you are delineating the ring path")
            return None

        if not two_dim_annotations and shape.shape_type != LabelmeShapeType.linestrip:
            st.error("Please delineate the path as a linestrip, where each points in the ring boundary")
            return None

        if shape.shape_type == LabelmeShapeType.line:
            return LineStringLabel(shape.points, label=shape.label)

        if shape.shape_type == LabelmeShapeType.linestrip:
            l_lines = [ LineString(np.concatenate((shape.points[idx].reshape((1,2)), shape.points[idx+1].reshape((1,2)))))
                       for idx in range(shape.points.shape[0]-1)]

            return MultiLineStringLabel(l_lines, label=shape.label)

        return None


    def compute_intersections(self, rings: LabelmeObject, path: LineString | MultiLineString, debug=True):
        if debug:
            image = load_image(self.read_file_path)
            debug_image_path = self.read_file_path.parent / f"debug_path.png"



        pith = True
        #compute intersections
        l_intersection = []
        flags = []
        colors = {}
        c = ColorCV2()

        for ring in rings.shapes:
            ring_polygon = Polygon(ring.points)
            if pith:
                centroid = ring_polygon.centroid
                pith = False
                l_intersection.append(PointLabelme(x=centroid.x, y=centroid.y, label="Pith"))

            if debug:
                rings_has_flag = len(ring.flags)>0
                if rings_has_flag:
                    exist_flag_within_list = len([f for f in flags if f == ring.flags['0']])
                    if not exist_flag_within_list:
                        flags.append(ring.flags['0'])
                        colors[ring.flags['0']] = c.get_next_color()

                    image = Drawing.curve(ring_polygon.exterior, image, color=colors[ring.flags['0']], thickness=2)
                else:
                    image = Drawing.curve(ring_polygon.exterior, image, color=c.red, thickness=2)

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
            write_image(str(debug_image_path), image)

        return l_intersection

    def compute_metrics(self, l_intersection: List, coorecorder_output_path: Path, unit: str,
                        csv_output_path = None, coorecorder_image_name = None, pixel_per_mm = 1)\
            -> pd.DataFrame:
        from lib.metrics import PathMetrics

        path = PathMetrics(l_intersection, scale=1 / pixel_per_mm, image_name=self.read_file_path.name, unit=unit)


        path.export_coorecorder_format( output_path = coorecorder_output_path,
                                        image_name= coorecorder_image_name , pixel_per_mm = pixel_per_mm)

        df = path.compute()
        df.to_csv(csv_output_path)
        return df


    def from_structure_to_labelme_shape(self, structure_list):
        pass

    def from_labelme_shape_to_structure(self, shapes):
        pass


########################################################################################################################

def main(runtime_config_path):
    ui = UI(runtime_config_path)
    if check_image(ui.CTX):
        return

    tab1, tab2 = st.tabs(["Area-Based", "Path-Based"])

    with tab1:
        #area based
        selected = ui.area_metrics()

    with tab2:
        #path based
        ui.path_metrics()

    st.divider()


    ui.CTX.save_config()
    return
