import os

import cv2
import streamlit as st
import numpy as np

from pathlib import Path
from copy import deepcopy

from lib.image import Drawing, load_image, write_image
from ui.common import Context, Shapes, file_uploader, RunningWidget, check_image
from lib.io import load_json, write_json
from backend.labelme_layer import LabelmeInterface as UserInterface, ring_relabelling



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
        self.autocomplete_ring_date = config["metadata"]["autocomplete_ring_date"]
        self.harvest_date = int(config["metadata"]["harvest_date"]["year"])
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
        self.other_annotation_file = config_manual["annotations_files"]["other"]
        self.other_annotation_file = None if len(self.other_annotation_file)==0 else self.other_annotation_file

        self.annotate_from_scratch = config_manual["annotate_from_scratch"]
        self.show_advanced_settings = config_manual["show_advanced_settings"]

        self.shape_visualization_settings = {}
        for shape in [Shapes.latewood, Shapes.earlywood, Shapes.other]:
            shape_visualization = config_manual["advanced_settings"][shape]
            shapes_list = shape_visualization["shapes_list"]
            thickness = shape_visualization["thickness"]
            color = shape_visualization["color"]
            #convert string "(R,G,B)" to integer tuple (R,G,B)
            color = tuple(map(int, color.strip('()').split(',')))
            stroke = shape_visualization["stroke"]
            fill = shape_visualization["fill"]
            opacity = shape_visualization["opacity"]

            self.shape_visualization_settings[shape] = VisualizationShape(shapes_list, thickness, color, stroke, fill,
                                                                          opacity)




        return

    def get_annotation_file_given_shape(self, shape):
        if shape == Shapes.earlywood:
            return self.ew_annotation_file
        if shape == Shapes.latewood:
            return self.lw_annotation_file
        if shape == Shapes.other:
            return self.other_annotation_file

    def update_config(self):
        config_manual = self.config["manual"]
        config_manual["main_shape"] = self.main_shape
        config_manual["drawable_shapes"] = self.drawable_shapes

        config_manual["annotations_files"]["early_wood"] =  [] if self.ew_annotation_file is None  \
            else str(self.ew_annotation_file)
        config_manual["annotations_files"]["late_wood"] = [] if self.lw_annotation_file is None \
            else str(self.lw_annotation_file)
        config_manual["annotations_files"]["other"] = [] if self.other_annotation_file is None \
            else str(self.other_annotation_file)


        config_manual["show_advanced_settings"] = self.show_advanced_settings
        for shape in [Shapes.latewood, Shapes.earlywood, Shapes.other]:
            shape_settings_memory = self.shape_visualization_settings[shape]
            shape_visualization_config = config_manual["advanced_settings"][shape]
            shape_visualization_config["shapes_list"] = shape_settings_memory.shapes_list
            shape_visualization_config["thickness"] = shape_settings_memory.thickness
            shape_visualization_config["color"] = str(shape_settings_memory.color)
            shape_visualization_config["stroke"] = shape_settings_memory.stroke
            shape_visualization_config["fill"] = shape_settings_memory.fill
            shape_visualization_config["opacity"] = shape_settings_memory.opacity

        config_manual["annotate_from_scratch"] = self.annotate_from_scratch

        return

    def reset_parameters(self):
        #TODO: Implement
        pass





class UI:

    def __init__(self, runtime_config_path):
        st.header("Ring Editing")
        st.markdown(
            """
            This interface allows you to edit the ring annotations. You can select the shape you want to edit.
            """
        )
        st.divider()
        CTX = ViewContext(runtime_config_path)
        CTX.init_specific_ui_components()
        self.CTX = CTX
        self.enabled_main_shape = False

    def main_shape(self):
        shapes_list = [Shapes.latewood, Shapes.earlywood, Shapes.other]
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

        drawable_knot = self.CTX.drawable_shapes[Shapes.other]
        with col3:
            knot_check_box = st.checkbox("Other", disabled= self.CTX.main_shape == Shapes.other, value=drawable_knot)
        self.CTX.drawable_shapes[Shapes.other] = knot_check_box


        return



    def is_bold(self, shape):
        return shape == self.CTX.main_shape
    def bold_text_depending_on_main_shape(self, shape, text):
        return f":red[**{text}**]" if self.is_bold(shape) else text

    def annotations_files(self):
        st.write("Upload the annotations files for the shapes you want to edit or visualize")
        #add checkbox
        annotate_from_scratch = st.checkbox("Annotate from scratch", self.CTX.annotate_from_scratch)
        if annotate_from_scratch:
            self.CTX.annotate_from_scratch = True
        else:
            self.CTX.annotate_from_scratch = False


        self.CTX.other_annotation_file = file_uploader( self.bold_text_depending_on_main_shape(Shapes.other,
                        f"Choose {Shapes.other} annotations file"),
                             self.CTX.output_dir / "other.json", "json")


        self.CTX.ew_annotation_file = file_uploader( self.bold_text_depending_on_main_shape(Shapes.earlywood,
                         f"Choose {Shapes.earlywood} annotations file"),
                              self.CTX.output_dir / "earlywood_read.json", "json", CTX=self.CTX)

        self.CTX.lw_annotation_file = file_uploader(self.bold_text_depending_on_main_shape(Shapes.latewood,
                            f"Choose {Shapes.latewood} annotations file"),
                                 self.CTX.output_dir / "latewood_read.json", "json", CTX=self.CTX)

        self.annotations_files_dict = { Shapes.earlywood: self.CTX.ew_annotation_file,
                                        Shapes.latewood: self.CTX.lw_annotation_file,
                                        Shapes.other: self.CTX.other_annotation_file
        }
        self.enabled_main_shape = (Path(self.annotations_files_dict[self.CTX.main_shape]).exists() or
                                   self.CTX.annotate_from_scratch)




        #return None

    def show_advanced_settings(self, value= None):
        st.write("Shapes Visualization Settings")
        show_advanced_settings = st.checkbox("Show advanced visualization settings", self.CTX.show_advanced_settings if value is None else value,
                                             disabled=not self.enabled_main_shape)
        self.CTX.show_advanced_settings = show_advanced_settings
        return show_advanced_settings

    @staticmethod
    def build_shapes_list(uploaded_other_annotation_file, uploaded_ew_annotation_file,
                          uploaded_lw_annotation_file):
        shapes_list = [Shapes.latewood, Shapes.earlywood, Shapes.other]
        if not uploaded_other_annotation_file:
            shapes_list.pop(shapes_list.index(Shapes.other))

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

        color_hex = '#%02x%02x%02x' % params.color
        color = st.color_picker("Pick A Color", color_hex, disabled=is_disabled)

        if not is_disabled:
            #convert color to BGR
            color = color.lstrip('#')
            color = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
            #switch to RGB
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
                              [Shapes.other]
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
        shapes_list = self.build_shapes_list(self.CTX.other_annotation_file,
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

        elif visualization_shape == Shapes.other:
            self.advanced_settings_shapes_visualization(params, is_disabled, Shapes.other)





    def edition(self):
        enabled = self.enabled_main_shape and self.CTX.output_dir is not None
        edit_button = st.button("Edit", disabled = not enabled)

        if edit_button:

            image_annotations_path = self.annotations_files_dict[self.CTX.main_shape]
            output_path = image_annotations_path
            if self.CTX.annotate_from_scratch:
                image_annotations_path = False
            else:
                output_name = self.CTX.main_shape.lower().replace(" ", "_") + "_backup.json"
                backup_path = self.CTX.output_dir / output_name
                os.system(f"cp {image_annotations_path} {backup_path}")
            if not self.CTX.image_path.exists():
                st.error("No image found. Please upload an imagen")
                return
            image_with_drawable_shapes_path = self.draw_shapes_over_image(self.CTX.image_path, self.CTX.drawable_shapes)
            gif_runner = RunningWidget()
            shape_edition = ShapeInterface(image_with_drawable_shapes_path, output_path, image_annotations_path)
            shape_edition.parse_input()
            shape_edition.interface()

            #relabel rings
            if self.CTX.autocomplete_ring_date:
                ring_relabelling(self.CTX.image_path, output_path, self.CTX.harvest_date)

            gif_runner.empty()
            st.write("Annotations saved in", output_path)
            st.warning("Remember, if you want to edit the annotations again, you'll need to upload the new annotations "
                       "file. The path to the new annotations file is displayed above. Otherwise, any edits to "
                       "previous annotations will be lost")



    def draw_shapes_over_image(self, image_path, drawable_shapes, output_image_name="images_with_shapes.png"):

        image = load_image(image_path)
        for shape in drawable_shapes:
            if drawable_shapes[shape] and not shape == self.CTX.main_shape:
                annotation_file = self.annotations_files_dict[shape]
                if not annotation_file:
                    continue
                #get visualization settings
                visualization_shape = self.CTX.shape_visualization_settings[shape]
                thickness = visualization_shape.thickness
                color = visualization_shape.color # getattr(ColorCV2, visualization_shape.color)
                color = color[::-1] #switch to BGR
                stroke = visualization_shape.stroke
                fill = visualization_shape.fill
                opacity = visualization_shape.opacity
                poly_shapes = UserInterface.load_shapes(annotation_file)
                for poly in poly_shapes:
                    image = Drawing.curve(poly.exterior.coords, image, color, thickness )

                if not fill:
                    continue
                #fill and opacity
                for poly in poly_shapes:
                    image = Drawing.fill(poly.exterior.coords, image, color, opacity)


        output_image_path = self.CTX.output_dir / output_image_name
        write_image(str(output_image_path), image)

        return output_image_path





########################################################################################################################
################################INTERFACE CLASSES#######################################################################
########################################################################################################################

class ShapeInterface(UserInterface):

    def __init__(self, image_path : Path, output_file : Path,  annotations_path : Path):
        super().__init__(read_file_path = image_path, write_file_path = output_file, edit = annotations_path)
        self.annotations_path = annotations_path


    def parse_input(self):
        try:
            data = load_json(self.annotations_path)

        except FileNotFoundError:
            st.write("No json file found")
            data = {}
        data["imagePath"] = str(self.read_file_path.name)
        write_json(data, self.write_file_path)
        return data


    def parse_output(self):
        try:
            data = load_json(self.write_file_path)
        except FileNotFoundError:
            st.write("No json file found")
            data = {}
        return data

    def from_structure_to_labelme_shape(self, structure_list):
        pass

    def from_labelme_shape_to_structure(self, shapes):
        pass

class PithBoundaryInterface(UserInterface):

    def __init__(self, image_path, output_file):
        super().__init__(read_file_path = image_path, write_file_path=output_file)

    def parse_output(self):
        try:
            data = load_json(self.output_path)
        except FileNotFoundError:
            st.write("No json file found")

        line = np.array(data['shapes'][0]['points'])
        pixels_length = int(np.linalg.norm(line[0] - line[1]))
        return pixels_length




########################################################################################################################
def main(runtime_config_path):
    ui = UI(runtime_config_path)
    if check_image(ui.CTX):
        return

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