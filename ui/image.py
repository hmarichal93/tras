import streamlit as st
import numpy as np
import cv2
import datetime
import os

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # O establece un límite más alto

from streamlit_option_menu import option_menu
from pathlib import Path

from lib.image import  write_image, remove_salient_object, resize_image
from ui.common import Context, RunningWidget, Pith, display_image
from backend.labelme_layer import (LabelmeShapeType,
                                   LabelmeObject, LabelmeInterface as UserInterface, resize_annotations,
                                   AL_LateWood_EarlyWood)


pixels_length = 1
know_distance = 2
mode = None
bg_image = None
bg_image_pil = None
bg_image_pil_no_background = None
image_path = "output/image.png"
image_no_background_path = "output/image_no_background.png"
background_json_path = "output/background.json"
DISPLAY_IMAGE_SIZE = 256

scale_json_file = "output/scale.json"
config = None

class ViewContext(Context):
    def init_specific_ui_components(self):
        config = self.config["image"]

        self.image_path = self.output_dir / config["image_path"]
        self.mode = config["mode"]

        self.scale_json_path = self.output_dir / config["scale"]["json_path"]
        self.units_mode = config["scale"]["unit"]
        self.pixels_length = config["scale"]["pixels_length"]
        self.know_distance = config["scale"]["know_distance"]
        self.dpi = config["scale"]["dpi"]
        self.scale_status = config["scale"]["status"]


        self.tree_planting_date = config["metadata"]["tree_planting_date"]
        self.harvest_date = config["metadata"]["harvest_date"]
        self.location = config["metadata"]["location"]
        self.species = config["metadata"]["species"]
        self.observations = config["metadata"]["observations"]
        self.code = config["metadata"]["code"]
        self.latitude = config['metadata']['latitude']
        self.longitude = config['metadata']['longitude']

        self.image_no_background_path = self.output_dir / config["background"]["image_path"]
        self.background_json_path = self.output_dir / config["background"]["json_path"]
        self.resize_factor = config["background"]["resize_factor"]

        #runtime variables
        self.bg_image = None
        self.bg_image_pil = None
        self.bg_image_pil_no_background = None


    def update_config(self):
        config = self.config["image"]

        config["mode"] = self.mode
        config["image_path"] = str(self.image_path.name)


        config["scale"]["unit"] = self.units_mode
        config["scale"]["pixels_length"] = self.pixels_length
        config["scale"]["know_distance"] = self.know_distance
        config["scale"]["dpi"] = self.dpi
        config["scale"]["status"] = self.scale_status


        config["metadata"]["tree_planting_date"] = self.tree_planting_date
        config["metadata"]["harvest_date"] = self.harvest_date
        config["metadata"]["location"] = self.location
        config["metadata"]['latitude'] = self.latitude
        config['metadata']['longitude'] = self.longitude
        config["metadata"]["species"] = self.species
        config["metadata"]["observations"] = self.observations
        config["metadata"]["code"] = self.code

        config["background"]["resize_factor"] = self.resize_factor

        return

        # config["background"]["image_path"] = self.image_no_background_path
        # config["background"]["json_path"] = self.background_json_path


class Menu:
    image = "Upload Image"
    scale = "Set Scale"
    preprocess  = "Preprocess"
    metadata = "Metadata"



def set_date_input(dictionary_date, text="Tree planting date"):
    year, month, day = (dictionary_date['year'], dictionary_date['month'],
                        dictionary_date['day'])
    input_date = st.date_input( text, datetime.date(year, month, day),
                                min_value = datetime.date(1500, 1, 1),
                                max_value = datetime.date(3000, 1, 1))
    dictionary_date['year'] = input_date.year
    dictionary_date['month'] = input_date.month
    dictionary_date['day'] = input_date.day
    return dictionary_date




def main(runtime_config_path):
    global CTX
    CTX = ViewContext(runtime_config_path)
    CTX.init_specific_ui_components()

    st.header("Image")
    st.markdown(
        """
        Upload Disk Image, preprocess, set scale unit and remove background
        """
    )

    selected = option_menu(None, [Menu.image, Menu.preprocess, Menu.scale,  Menu.metadata],
                           menu_icon="cast", default_index=0, orientation="horizontal")



    if selected == Menu.image:
        # CTX.mode = st.radio("Mode", ("cross-section", "core"), horizontal=True, index = 0 if CTX.mode == "cross-section"
        #                                         else 1 )

        CTX.bg_image = st.file_uploader("Image:", type=["png", "jpg"])
        gif_runner = RunningWidget()
        if CTX.bg_image is not None:
            os.system(f"rm -rf {CTX.output_dir}/*.png")
            os.system(f"rm -rf {CTX.output_dir}/*.json")
            os.system(f"rm -rf {CTX.output_dir}/metrics")
            Path(CTX.output_dir).mkdir(parents=True, exist_ok=True)
            CTX.scale_status = False

            CTX.bg_image_pil = Image.open(CTX.bg_image)
            CTX.bg_image_pil.save(CTX.image_path)
            bg_image_pil_display = CTX.bg_image_pil.resize((CTX.display_image_size, CTX.display_image_size),
                                                           Image.Resampling.LANCZOS)
            #st.image(bg_image_pil_display)
            display_image(bg_image_pil_display)

        elif Path(CTX.image_path).exists():
            CTX.bg_image_pil = Image.open(CTX.image_path)
            bg_image_pil_display = CTX.bg_image_pil.resize((CTX.display_image_size, CTX.display_image_size),
                                                           Image.Resampling.LANCZOS)
            #st.image(bg_image_pil_display)
            display_image(bg_image_pil_display)
        gif_runner.empty()


    if selected == Menu.preprocess and Path(CTX.image_path).exists():
        radio = st.radio("Remove Background", (Pith.manual, Pith.automatic), index=0, horizontal=True)

        if st.button("Remove Background"):
            gif_runner = RunningWidget()
            interface = BackgroundInterface(CTX.image_path, CTX.background_json_path, CTX.image_no_background_path)
            if radio == Pith.manual:
                interface.interface()
            else:
                interface.automatic()
            res = interface.parse_output()
            if res is not None:
                CTX.bg_image_pil_no_background = interface.remove_background()
                CTX.bg_image_pil_no_background = CTX.bg_image_pil_no_background.resize((CTX.display_image_size,
                                                                CTX.display_image_size), Image.Resampling.LANCZOS)

            gif_runner.empty()

        if Path(CTX.image_no_background_path).exists():
            CTX.bg_image_pil_no_background = Image.open(CTX.image_no_background_path)
            CTX.bg_image_pil_no_background = CTX.bg_image_pil_no_background.resize((CTX.display_image_size,
                                                                                    CTX.display_image_size),
                                                                                   Image.Resampling.LANCZOS)
            #st.image(CTX.bg_image_pil_no_background)
            display_image(CTX.bg_image_pil_no_background)

        resize_factor = st.slider("Resize Factor", 0.0, 10.0, CTX.resize_factor , help="Resize factor for the image.\n"
                                                                                       "Be aware that the image will \n"
                                                                                       "be resized, which means that metrics\n"
                                                                                       "will be affected")
        if resize_factor != CTX.resize_factor and resize_factor > 0:
            CTX.resize_factor = resize_factor

        if st.button("Resize Image"):
            gif_runner = RunningWidget()

            _ = resize_image(CTX.image_path, resize_factor)

            if Path(CTX.image_no_background_path).exists():
                backup_image_path = str(CTX.image_no_background_path).replace(".png", "_bkp.png")
                os.system(f"cp {CTX.image_no_background_path} {backup_image_path}")
                CTX.image_no_background_path = resize_image(CTX.image_no_background_path, resize_factor)
                resize_image(str(CTX.image_no_background_path).replace(".png", "_mask.png"), resize_factor)
                new_annotations_path = resize_annotations(backup_image_path, CTX.image_no_background_path, CTX.background_json_path)
                os.system(f"cp {new_annotations_path} {CTX.background_json_path}")


            CTX.scale_status = False
            gif_runner.empty()
            st.warning("Image resized. Please set the scale again")

    if selected == Menu.scale and Path(CTX.image_path).exists():
        CTX.units_mode = st.radio(
            "Unit:",
            ("nm", r"$\mu$m", "mm", "cm", "dpi"), horizontal=True, index=scale_index_unit(CTX.units_mode)
        )
        if CTX.units_mode == "dpi":
            dpi = st.number_input("DPI scale:", 1, 2000, CTX.dpi)
            if CTX.dpi != dpi:
                CTX.scale_status = True
                CTX.dpi = dpi

        else:
            button = st.button("Set Distance in Pixels")
            if button:
                gif_runner = RunningWidget()
                CTX.pixels_length = set_scale(CTX)
                CTX.scale_status = True
                gif_runner.empty()
            pixels_length = st.number_input("Distance in Pixels", 1.0, 10000.0, float(CTX.pixels_length))
            #input float number

            if pixels_length != CTX.pixels_length:
                CTX.scale_status = True
                CTX.pixels_length = pixels_length
            know_distance = st.number_input("Know distance", 1, 100, CTX.know_distance)
            if know_distance != CTX.know_distance:
                CTX.scale_status = True
                CTX.know_distance = know_distance

    if selected == Menu.metadata and Path(CTX.image_path).exists():

        code = st.text_input("Code", value = CTX.code)
        CTX.code = code

        ##tree planting date
        CTX.tree_planting_date = set_date_input(CTX.tree_planting_date, "Tree planting date")
        CTX.harvest_date = set_date_input(CTX.tree_planting_date, "Harvest date")

        location = st.text_input("Location", value = CTX.location)
        CTX.location = location

        latitude = st.number_input("Latitude", value = CTX.latitude, format="%.8f")
        CTX.latitude = latitude

        longitude = st.number_input("Longitude", value = CTX.longitude, format="%.8f")
        CTX.longitude = longitude

        species = st.text_input("Species", value = CTX.species)
        CTX.species = species

        observations = st.text_area("Observations", value = CTX.observations)
        CTX.observations = observations


    #save status
    CTX.save_config()

def scale_index_unit(unit):
    if unit == "cm":
        return 3
    elif unit == "mm":
        return 2
    elif unit == r"$\mu$m":
        return 1
    elif unit == r"nm":
        return 0
    else:
        return 4




########################################################################################################################
################################INTERFACE CLASSES#######################################################################
########################################################################################################################


class BackgroundInterface(UserInterface):
    def __init__(self, image_path, output_json_path, output_image_path):
        super().__init__(read_file_path = image_path, write_file_path=output_json_path)
        self.output_image_path = output_image_path

    def parse_output(self):
        object = LabelmeObject(self.write_file_path)
        if len(object.shapes) > 1:
            st.error("More than one shape found. Add only one shape")
            return None
        shape = object.shapes[0]
        if not(shape.shape_type == LabelmeShapeType.polygon):
            st.error("Shape is not a polyline. Remember that you are delineating the disk contour")
            return None

        self.background_polygon = shape.points[:, [1,0]]
        return  self.background_polygon

    def remove_background(self):
        bg_image_pil = Image.open(self.read_file_path)
        bg_image_pil_no_background, mask = self.remove_background_polygon(bg_image_pil, self.background_polygon)
        bg_image_pil_no_background.save(self.output_image_path)
        write_image(str(self.output_image_path).replace(".png", "_mask.png"), mask)
        return bg_image_pil_no_background

    def remove_background_polygon(self, bg_image_pil, background_polygon):
        "given a polygon, remove the background of the image"
        bg_image_pil = bg_image_pil.convert("RGB")
        bg_image = np.array(bg_image_pil)
        # create mask
        mask = np.zeros(bg_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [background_polygon.astype(int)], (255))
        # apply mask
        bg_image[mask != 255] = [255, 255, 255]
        bg_image_pil = Image.fromarray(bg_image)
        #invert mask values
        mask = cv2.bitwise_not(mask)
        return bg_image_pil, mask


    def from_structure_to_labelme_shape(self, structure_list):
        pass

    def from_labelme_shape_to_structure(self, shapes):
        pass

    def automatic(self):
        disk_mask = remove_salient_object(self.read_file_path, self.output_image_path)
        # Ensure the mask is binary (values 0 and 255 only)
        _, binary_mask = cv2.threshold(disk_mask, 127, 255, cv2.THRESH_BINARY)

        # Find the contours in the binary mask
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        al = AL_LateWood_EarlyWood(None,
                                   write_file_path=str(self.write_file_path),
                                   image_path=str(self.read_file_path)
                                   )
        al.write_list_of_points_to_labelme_json([largest_contour.squeeze()[:,[1,0]].tolist()])



class ScaleInterface(UserInterface):

    def __init__(self, image_path, output_file):
        super().__init__(read_file_path = image_path, write_file_path=output_file)

    def parse_output(self):
        object = LabelmeObject(self.write_file_path)
        if len(object.shapes) > 1:
            st.error("More than one shape found. Add only one shape")
            return None
        shape = object.shapes[0]
        if not(shape.shape_type == LabelmeShapeType.line):
            st.error("Shape is not a line. Remember that you are marking the scale")
            return 1
        line = np.array(shape.points)
        pixels_length = int(np.linalg.norm(line[0] - line[1]))
        return pixels_length

    def from_structure_to_labelme_shape(self, structure_list):
        pass

    def from_labelme_shape_to_structure(self, shapes):
        pass


def set_scale(CTX):
    scale = ScaleInterface(CTX.image_path, CTX.scale_json_path)
    scale.interface()
    CTX.pixels_length = scale.parse_output()

    return CTX.pixels_length