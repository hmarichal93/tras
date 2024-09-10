import os

import streamlit as st
import numpy as np

from streamlit_option_menu import option_menu
from pathlib import Path

from lib.image import LabelMeInterface as UserInterface
from ui.common import Context, Shapes, Color
from lib.io import load_json, write_json, bytesio_to_dict




def annotate_pith():


    return


class ViewContext(Context):
    def init_specific_ui_components(self):
        config = self.config["image"]

        self.image_path = self.output_dir / config["image_path"]
        self.image_no_background_path = self.output_dir / config["background"]["image_path"]
        if Path(self.image_no_background_path).exists():
            self.image_path = self.image_no_background_path


        config_manual = self.config["manual"]

        self.main_shape = config_manual["main_shape"]

        self.drawable_shapes = config_manual["drawable_shapes"]

        self.ew_annotation_file = config_manual["annotations_files"]["early_wood"]
        self.ew_annotation_file = None if len(self.ew_annotation_file)==0 else self.ew_annotation_file
        self.lw_annotation_file = config_manual["annotations_files"]["late_wood"]
        self.lw_annotation_file = None if len(self.lw_annotation_file)==0 else self.lw_annotation_file
        self.knot_annotation_file = config_manual["annotations_files"]["knot"]
        self.knot_annotation_file = None if len(self.knot_annotation_file)==0 else self.knot_annotation_file
        self.cw_annotation_file = config_manual["annotations_files"]["compression_wood"]
        self.cw_annotation_file = None if len(self.cw_annotation_file)==0 else self.cw_annotation_file

        self.show_advanced_settings = config_manual["show_advanced_settings"]

        self.shapes_list = config_manual["advanced_settings"]["shapes_list"]
        self.thickness = config_manual["advanced_settings"]["thickness"]
        self.color = config_manual["advanced_settings"]["color"]
        self.stroke = config_manual["advanced_settings"]["stroke"]
        self.fill = config_manual["advanced_settings"]["fill"]
        self.opacity = config_manual["advanced_settings"]["opacity"]



        return

    def update_config(self):
        config_manual = self.config["manual"]
        config_manual["main_shape"] = self.main_shape
        config_manual["drawable_shapes"] = self.drawable_shapes

        config_manual["annotations_files"]["early_wood"] =  [] if self.ew_annotation_file is None  \
            else str(self.ew_annotation_file)
        config_manual["annotations_files"]["late_wood"] = [] if self.lw_annotation_file is None \
            else str(self.lw_annotation_file)
        config_manual["annotations_files"]["knot"] = [] if self.knot_annotation_file is None \
            else str(self.knot_annotation_file)
        config_manual["annotations_files"]["compression_wood"] = [] if self.cw_annotation_file is None \
            else str(self.cw_annotation_file)

        config_manual["show_advanced_settings"] = self.show_advanced_settings

        config_manual["advanced_settings"]["shapes_list"] = self.shapes_list
        config_manual["advanced_settings"]["thickness"] = self.thickness
        config_manual["advanced_settings"]["color"] = self.color
        config_manual["advanced_settings"]["stroke"] = self.stroke
        config_manual["advanced_settings"]["fill"] = self.fill
        config_manual["advanced_settings"]["opacity"] = self.opacity

        return

    def reset_parameters(self):
        #TODO: Implement
        pass


class ShapeInterface(UserInterface):

    def __init__(self, image_path, output_file,  annotations_path):
        super().__init__(image_path, output_file, edit=annotations_path)
        self.annotations_path = annotations_path


    def parse_input(self):
        try:
            data = load_json(self.annotations_path)

        except FileNotFoundError:
            st.write("No json file found")
            data = {}
        data["imagePath"] = str(self.image_path.name)
        write_json(data, self.output_path)
        return data






    def parse_output(self):
        try:
            data = load_json(self.output_path)
        except FileNotFoundError:
            st.write("No json file found")
            data = {}
        return data





class UI:

    def __init__(self, runtime_config_path):
        CTX = ViewContext(runtime_config_path)
        CTX.init_specific_ui_components()
        self.CTX = CTX
        self.enabled_main_shape = False

    def main_shape(self):
        shapes_list = [Shapes.latewood, Shapes.earlywood, Shapes.knot, Shapes.compresionwood]
        selected = st.radio("Select shape to annotate or edit", shapes_list ,
                            horizontal=True, index= shapes_list.index(self.CTX.main_shape))
        self.CTX.main_shape = selected
        return selected

    def drawable_shapes(self):
        st.write("Select the extra shapes you want to draw over image")
        col1, col2, col3, col4 = st.columns(4)
        drawable_ew = self.CTX.drawable_shapes[Shapes.earlywood]
        with col1:
            ew_check_box = st.checkbox("Earlywood", disabled= self.CTX.main_shape == Shapes.earlywood, value=drawable_ew)
        self.CTX.drawable_shapes[Shapes.earlywood] = ew_check_box

        drawable_lw = self.CTX.drawable_shapes[Shapes.latewood]
        with col2:
            lw_check_box = st.checkbox("Latewood", disabled= self.CTX.main_shape == Shapes.latewood, value=drawable_lw)
        self.CTX.drawable_shapes[Shapes.latewood] = lw_check_box

        drawable_knot = self.CTX.drawable_shapes[Shapes.knot]
        with col3:
            knot_check_box = st.checkbox("Knot", disabled= self.CTX.main_shape == Shapes.knot, value=drawable_knot)
        self.CTX.drawable_shapes[Shapes.knot] = knot_check_box

        drawable_cw = self.CTX.drawable_shapes[Shapes.compresionwood]
        with col4:
            cw_check_box = st.checkbox("Compression Wood", disabled= self.CTX.main_shape == Shapes.compresionwood,
                                       value=drawable_cw)
        self.CTX.drawable_shapes[Shapes.compresionwood] = cw_check_box

        return

    @staticmethod
    def save_annotation_file_locally(filename, file_uploader_instance):
        config = bytesio_to_dict(file_uploader_instance)
        write_json(config, filename)

    def is_bold(self, shape):
        return shape == self.CTX.main_shape
    def bold_text_depending_on_main_shape(self, shape, text):
        return f":red[**{text}**]" if self.is_bold(shape) else text

    def annotations_files(self):
        st.write("Upload the annotations files for the shapes you want to edit or visualize")


        self.CTX.knot_annotation_file = self.file_uploader( self.bold_text_depending_on_main_shape(Shapes.knot,
                        f"Choose {Shapes.knot} annotations file"),
                             self.CTX.output_dir / "knot.json", "json")

        self.CTX.cw_annotation_file = self.file_uploader(self.bold_text_depending_on_main_shape(Shapes.compresionwood,
                        f"Choose {Shapes.compresionwood} annotations file"),
                             self.CTX.output_dir / "compressionwood.json", "json")

        self.CTX.ew_annotation_file = self.file_uploader( self.bold_text_depending_on_main_shape(Shapes.earlywood,
                         f"Choose {Shapes.earlywood} annotations file"),
                              self.CTX.output_dir / "earlywood.json", "json")

        self.CTX.lw_annotation_file = self.file_uploader(self.bold_text_depending_on_main_shape(Shapes.latewood,
                            f"Choose {Shapes.latewood} annotations file"),
                                 self.CTX.output_dir / "latewood.json", "json")

        self.annotations_files_dict = { Shapes.earlywood: self.CTX.ew_annotation_file,
                                        Shapes.latewood: self.CTX.lw_annotation_file,
                                        Shapes.knot: self.CTX.knot_annotation_file,
                                        Shapes.compresionwood : self.CTX.cw_annotation_file
        }
        self.enabled_main_shape = self.annotations_files_dict[self.CTX.main_shape] is not None

    def file_uploader(self, label, output_file, extension):
        uploaded_cw_annotation_file = st.file_uploader(label, type=[extension])
        if uploaded_cw_annotation_file:
            self.save_annotation_file_locally(output_file, uploaded_cw_annotation_file)
            return output_file
        return None

    def show_advanced_settings(self):
        st.write("Shapes Visualization Settings")
        show_advanced_settings = st.checkbox("Show advanced visualization settings", self.CTX.show_advanced_settings,
                                             disabled=not self.enabled_main_shape)
        self.CTX.show_advanced_settings = show_advanced_settings
        return show_advanced_settings

    @staticmethod
    def build_shapes_list(uploaded_cw_annotation_file, uploaded_knot_annotation_file, uploaded_ew_annotation_file,
                          uploaded_lw_annotation_file):
        shapes_list = [Shapes.latewood, Shapes.earlywood, Shapes.knot, Shapes.compresionwood]
        if not uploaded_cw_annotation_file:
            shapes_list.pop(shapes_list.index(Shapes.compresionwood))

        if not uploaded_knot_annotation_file:
            shapes_list.pop(shapes_list.index(Shapes.knot))

        if not uploaded_ew_annotation_file:
            shapes_list.pop(shapes_list.index(Shapes.earlywood))

        if not uploaded_lw_annotation_file:
            shapes_list.pop(shapes_list.index(Shapes.latewood))

        return shapes_list

    def default_visualization_params(self):
        params = {}
        params["thickness"] = self.CTX.thickness
        params["color"] = self.CTX.color
        params["stroke"] = self.CTX.stroke
        params["fill"] = self.CTX.fill
        params["opacity"] = self.CTX.opacity
        return params
    def advanced_settings(self):
        shapes_list = self.build_shapes_list(self.CTX.cw_annotation_file, self.CTX.knot_annotation_file,
                                        self.CTX.ew_annotation_file, self.CTX.lw_annotation_file)

        params = self.default_visualization_params()

        shape_visualization_settings = {}
        for shape in shapes_list:
            shape_visualization_settings[shape] = params.copy()

        visualization_shape = st.selectbox( "Select Shape", shapes_list)
        is_disabled = visualization_shape == self.CTX.main_shape or len(shapes_list) == 0
        if not is_disabled:
            params = shape_visualization_settings[visualization_shape].copy()

        thickness = st.number_input("Thickness", 1, 100, params['thickness'],
                                    disabled = is_disabled)
        if not is_disabled:
            params['thickness'] = thickness

        color_list = [Color.red, Color.black, Color.white, Color.blue, Color.green, Color.yellow]
        color = st.radio("Color", color_list,
                         index=color_list.index(params["color"]),
                         horizontal=True, disabled = is_disabled)
        if not is_disabled:
            params['color'] = color

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            stroke = st.checkbox("Stroke", params['stroke'],
                                 disabled = is_disabled)
            if not is_disabled:
                params['stroke'] = stroke

        with col2:
            fill = st.checkbox("Fill", params['fill'],
                               disabled = is_disabled)
            if not is_disabled:
                params['fill'] = fill

        opacity = st.slider("Opacity", 0.0, 1.0, params['opacity'],
                            disabled = not fill or is_disabled)

        if not fill and not is_disabled:
            params['opacity'] = opacity

        if not is_disabled:
            shape_visualization_settings[visualization_shape] = params

        self.CTX.shapes_list = shapes_list
        self.CTX.thickness = params['thickness']
        self.CTX.color = params['color']
        self.CTX.stroke = params['stroke']
        self.CTX.fill = params['fill']
        self.CTX.opacity = params['opacity']

    def edition(self):
        enabled = self.enabled_main_shape and self.CTX.output_dir is not None
        edit_button = st.button("Edit", disabled = not enabled)

        if edit_button:
            output_name = self.CTX.main_shape.lower().replace(" ", "_") + "_edited.json"
            output_path = self.CTX.output_dir / output_name
            image_annotations_path = self.annotations_files_dict[self.CTX.main_shape]

            shape_edition = ShapeInterface(self.CTX.image_path, output_path, image_annotations_path)
            shape_edition.parse_input()
            shape_edition.interface()
            st.write("Annotations saved in", output_path)


            



def main(runtime_config_path):
    ui = UI(runtime_config_path)

    st.divider()
    selected = ui.main_shape()
    st.divider()

    ui.drawable_shapes()
    st.divider()

    ui.annotations_files()
    st.divider()

    show_advanced_settings = ui.show_advanced_settings()
    if show_advanced_settings:
        ui.advanced_settings()
    st.divider()

    ui.edition()
    st.divider()

    ui.CTX.save_config()

    return

def annotate_pith():
    #TODO: Implement
    pass




class PithBoundaryInterface(UserInterface):

    def __init__(self, image_path, output_file):
        super().__init__(image_path, output_file)

    def parse_output(self):
        try:
            data = load_json(self.output_path)
        except FileNotFoundError:
            st.write("No json file found")

        line = np.array(data['shapes'][0]['points'])
        pixels_length = int(np.linalg.norm(line[0] - line[1]))
        return pixels_length