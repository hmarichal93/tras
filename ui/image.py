import streamlit as st
import numpy as np
import cv2
import datetime

from PIL import Image
from streamlit_option_menu import option_menu
from pathlib import Path

from lib.image import LabelMeInterface as UserInterface
from lib.io import load_json
from ui.common import Context


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


        self.tree_planting_date = config["metadata"]["tree_planting_date"]
        self.location = config["metadata"]["location"]
        self.species = config["metadata"]["species"]
        self.observations = config["metadata"]["observations"]
        self.code = config["metadata"]["code"]

        self.image_no_background_path = self.output_dir / config["background"]["image_path"]
        self.background_json_path = self.output_dir / config["background"]["json_path"]

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


        config["metadata"]["tree_planting_date"] = self.tree_planting_date
        config["metadata"]["location"] = self.location
        config["metadata"]["species"] = self.species
        config["metadata"]["observations"] = self.observations
        config["metadata"]["code"] = self.code

        return

        # config["background"]["image_path"] = self.image_no_background_path
        # config["background"]["json_path"] = self.background_json_path


class Menu:
    image = "Upload Image"
    scale = "Set Scale"
    background  = "Remove Background"
    metadata = "Metadata"


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

    selected = option_menu(None, [Menu.image, Menu.scale, Menu.background, Menu.metadata],
                           menu_icon="cast", default_index=0, orientation="horizontal")



    if selected == Menu.image:
        CTX.mode = st.radio("Mode", ("cross-section", "core"), horizontal=True, index = 0 if CTX.mode == "cross-section"
                                                else 1 )

        CTX.bg_image = st.file_uploader("Image:", type=["png", "jpg"])
        if CTX.bg_image:
            CTX.bg_image_pil = Image.open(CTX.bg_image)
            CTX.bg_image_pil.save(CTX.image_path)
            bg_image_pil_display = CTX.bg_image_pil.resize((CTX.display_image_size, CTX.display_image_size),
                                                           Image.Resampling.LANCZOS)
            st.image(bg_image_pil_display)

        elif Path(CTX.image_path).exists():
            CTX.bg_image_pil = Image.open(CTX.image_path)
            bg_image_pil_display = CTX.bg_image_pil.resize((CTX.display_image_size, CTX.display_image_size),
                                                           Image.Resampling.LANCZOS)
            st.image(bg_image_pil_display)


    if selected == Menu.scale and Path(CTX.image_path).exists():
        CTX.units_mode = st.radio(
            "Unit:",
            ("cm", "mm", "dpi"), horizontal=True, index = scale_index_unit(CTX.units_mode)
        )
        if CTX.units_mode == "dpi":
            CTX.dpi = st.number_input("DPI scale:", 1, 2000, CTX.dpi)

        else:
            button = st.button("Set Distance in Pixels")
            if button:
                CTX.pixels_length  = set_scale(CTX)
            CTX.pixels_length = st.number_input("Distance in Pixels", 1, 10000, CTX.pixels_length)
            CTX.know_distance = st.number_input("Know distance", 1, 100, CTX.know_distance)

    if selected == Menu.background and Path(CTX.image_path).exists():
        if st.button("Remove Background"):
            interface = BackgroundInterface(CTX.image_path, CTX.background_json_path, CTX.image_no_background_path)
            interface.interface()
            res = interface.parse_output()
            if res is not None:
                CTX.bg_image_pil_no_background = interface.remove_background()
                CTX.bg_image_pil_no_background = CTX.bg_image_pil_no_background.resize((CTX.display_image_size,
                                                                CTX.display_image_size), Image.Resampling.LANCZOS)
                st.image(CTX.bg_image_pil_no_background)

        if Path(CTX.image_no_background_path).exists():
            CTX.bg_image_pil_no_background = Image.open(CTX.image_no_background_path)
            CTX.bg_image_pil_no_background = CTX.bg_image_pil_no_background.resize((CTX.display_image_size,
                                                                                    CTX.display_image_size),
                                                                                   Image.Resampling.LANCZOS)
            st.image(CTX.bg_image_pil_no_background)

    if selected == Menu.metadata and Path(CTX.image_path).exists():

        code = st.text_input("Code", value = CTX.code)
        CTX.code = code

        year, month, day = (CTX.tree_planting_date['year'], CTX.tree_planting_date['month'],
                            CTX.tree_planting_date['day'])
        input_date = st.date_input( "Tree planting date", datetime.date(year, month, day),
                                    min_value = datetime.date(1500, 1, 1),
                                    max_value = datetime.date(3000, 1, 1))
        CTX.tree_planting_date = {}
        CTX.tree_planting_date['year'] = input_date.year
        CTX.tree_planting_date['month'] = input_date.month
        CTX.tree_planting_date['day'] = input_date.day


        location = st.text_input("Location", value = CTX.location)
        CTX.location = location

        species = st.text_input("Species", value = CTX.species)
        CTX.species = species

        observations = st.text_area("Observations", value = CTX.observations)
        CTX.observations = observations






    #save status
    CTX.save_config()

def scale_index_unit(unit):
    if unit == "cm":
        return 0
    elif unit == "mm":
        return 1
    else:
        return 2


class BackgroundInterface(UserInterface):
    def __init__(self, image_path, output_json_path, output_image_path):
        super().__init__(image_path, output_json_path)
        self.output_image_path = output_image_path

    def parse_output(self):
        try:
            data = load_json(self.output_path)
        except FileNotFoundError:
            st.write("No json file found")
            return None

        self.background_polygon = np.array(data['shapes'][0]['points'])
        return  self.background_polygon

    def remove_background(self):
        bg_image_pil = Image.open(self.image_path)
        bg_image_pil_no_background = self.remove_background_polygon(bg_image_pil, self.background_polygon)
        bg_image_pil_no_background.save(self.output_image_path)
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
        return bg_image_pil







class ScaleInterface(UserInterface):

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


def set_scale(CTX):
    scale = ScaleInterface(CTX.image_path, CTX.scale_json_path)
    scale.interface()
    CTX.pixels_length = scale.parse_output()

    return CTX.pixels_length