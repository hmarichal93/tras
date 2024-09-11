import os

import cv2
import streamlit as st
import numpy as np

from streamlit_option_menu import option_menu
from pathlib import Path
from copy import deepcopy

from lib.image import LabelMeInterface as UserInterface, Color as ColorCV2, Drawing, load_image
from ui.common import Context, Shapes, Color
from lib.io import load_json, write_json, bytesio_to_dict





class VisualizationShape:
    def __init__(self, shapes_list, thickness, color, stroke, fill, opacity):
        self.shapes_list = shapes_list
        self.thickness = thickness
        self.color = color
        self.stroke = stroke
        self.fill = fill
        self.opacity = opacity

    def __str__(self):
        return f"({self.thickness=} {self.color=} {self.stroke=} {self.fill=} {self.opacity=})"

    def __repr__(self):
        return f"({self.thickness=} {self.color=} {self.stroke=} {self.fill=} {self.opacity=})"

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

        self.shape_visualization_settings = {}
        for shape in [Shapes.latewood, Shapes.earlywood, Shapes.knot, Shapes.compresionwood]:
            shape_visualization = config_manual["advanced_settings"][shape]
            shapes_list = shape_visualization["shapes_list"]
            thickness = shape_visualization["thickness"]
            color = shape_visualization["color"]
            stroke = shape_visualization["stroke"]
            fill = shape_visualization["fill"]
            opacity = shape_visualization["opacity"]

            self.shape_visualization_settings[shape] = VisualizationShape(shapes_list, thickness, color, stroke, fill,
                                                                          opacity)




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
        for shape in [Shapes.latewood, Shapes.earlywood, Shapes.knot, Shapes.compresionwood]:
            shape_settings_memory = self.shape_visualization_settings[shape]
            shape_visualization_config = config_manual["advanced_settings"][shape]
            shape_visualization_config["shapes_list"] = shape_settings_memory.shapes_list
            shape_visualization_config["thickness"] = shape_settings_memory.thickness
            shape_visualization_config["color"] =  shape_settings_memory.color
            shape_visualization_config["stroke"] = shape_settings_memory.stroke
            shape_visualization_config["fill"] = shape_settings_memory.fill
            shape_visualization_config["opacity"] = shape_settings_memory.opacity


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

    def show_advanced_settings(self, value= None):
        st.write("Shapes Visualization Settings")
        show_advanced_settings = st.checkbox("Show advanced visualization settings", self.CTX.show_advanced_settings if value is None else value,
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
        params["thickness"] = self.CTX.shape_visualization_settings[self.CTX.main_shape].thickness
        params["color"] = self.CTX.shape_visualization_settings[self.CTX.main_shape].color
        params["stroke"] = self.CTX.shape_visualization_settings[self.CTX.main_shape].stroke
        params["fill"] = self.CTX.shape_visualization_settings[self.CTX.main_shape].fill
        params["opacity"] = self.CTX.shape_visualization_settings[self.CTX.main_shape].opacity
        return params

    def advanced_settings_shapes_visualization(self, params, is_disabled, visualization_shape):
        thickness = st.number_input("Thickness", 1, 100, params.thickness,
                                    disabled = is_disabled)
        if not is_disabled:
            params.thickness = thickness

        color_list = [Color.red, Color.black, Color.white, Color.blue, Color.green, Color.yellow]
        color = st.radio("Color", color_list,
                         index=color_list.index(params.color),
                         horizontal=True, disabled = is_disabled)
        if not is_disabled:
            params.color = color

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            stroke = st.checkbox("Stroke", params.stroke,
                                 disabled = is_disabled)
            if not is_disabled:
                params.stroke = stroke

        with col2:
            fill = st.checkbox("Fill", params.fill,
                               disabled = is_disabled and not visualization_shape in
                              [Shapes.compresionwood, Shapes.knot]
                   )
            if not is_disabled:
                params.fill = fill

        opacity = st.slider("Opacity", 0.0, 1.0, params.opacity,
                            disabled = not fill or is_disabled)

        if fill and not is_disabled:
            params.opacity = opacity

        if not is_disabled:
            self.CTX.shape_visualization_settings[visualization_shape] = params
    def advanced_settings(self):
        ###
        shapes_list = self.build_shapes_list(self.CTX.cw_annotation_file, self.CTX.knot_annotation_file,
                                        self.CTX.ew_annotation_file, self.CTX.lw_annotation_file)

        visualization_shape = st.selectbox( "Select Shape", shapes_list)
        is_disabled = visualization_shape == self.CTX.main_shape or len(shapes_list) == 0 or visualization_shape is None

        params = deepcopy(self.CTX.shape_visualization_settings[visualization_shape]) if \
                        not is_disabled else\
                 deepcopy(self.CTX.shape_visualization_settings[self.CTX.main_shape])

        if visualization_shape == Shapes.latewood:
            self.advanced_settings_shapes_visualization(params, is_disabled, Shapes.latewood)
        elif visualization_shape == Shapes.earlywood:
            self.advanced_settings_shapes_visualization(params, is_disabled, Shapes.earlywood)

        elif visualization_shape == Shapes.knot:
            self.advanced_settings_shapes_visualization(params, is_disabled, Shapes.knot)

        elif visualization_shape == Shapes.compresionwood:
            self.advanced_settings_shapes_visualization(params, is_disabled, Shapes.compresionwood)





    def edition(self):
        enabled = self.enabled_main_shape and self.CTX.output_dir is not None
        edit_button = st.button("Edit", disabled = not enabled)
        output_name = self.CTX.main_shape.lower().replace(" ", "_") + "_edited.json"
        output_path = self.CTX.output_dir / output_name
        if edit_button:
            image_annotations_path = self.annotations_files_dict[self.CTX.main_shape]
            image_with_drawable_shapes_path = self.draw_shapes_over_image(self.CTX.image_path, self.CTX.drawable_shapes)
            shape_edition = ShapeInterface(image_with_drawable_shapes_path, output_path, image_annotations_path)
            shape_edition.parse_input()
            shape_edition.interface()
            st.write("Annotations saved in", output_path)

    def draw_shapes_over_image(self, image_path, drawable_shapes, output_image_name="images_with_shapes.png"):
        image = load_image(image_path)
        for shape in drawable_shapes:
            if drawable_shapes[shape]:
                if not self.CTX.cw_annotation_file:
                    continue
                #get visualization settings
                visualization_shape = self.CTX.shape_visualization_settings[shape]
                thickness = visualization_shape.thickness
                color = getattr(ColorCV2, visualization_shape.color)
                stroke = visualization_shape.stroke
                fill = visualization_shape.fill
                opacity = visualization_shape.opacity
                poly_shapes = UserInterface.load_shapes(self.CTX.cw_annotation_file)
                for poly in poly_shapes:
                    image = Drawing.curve(poly.exterior.coords, image, color, thickness )

                if not fill:
                    continue
                #fill and opacity
                for poly in poly_shapes:
                    image = Drawing.fill(poly.exterior.coords, image, color, opacity)


        output_image_path = self.CTX.output_dir / output_image_name
        cv2.imwrite(str(output_image_path), image)

        return output_image_path



def main(runtime_config_path):
    ui = UI(runtime_config_path)

    st.divider()
    selected = ui.main_shape()
    st.divider()

    ui.annotations_files()
    st.divider()

    ui.drawable_shapes()
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