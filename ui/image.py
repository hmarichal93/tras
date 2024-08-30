import streamlit as st
import numpy as np
import cv2

from PIL import Image
from streamlit_option_menu import option_menu

from lib.image import LabelMeInterface as UserInterface
from lib.io import load_json

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


class Menu:
    image = "Upload Image"
    scale = "Set Scale"
    background  = "Remove Background"


def main():
    global pixels_length, know_distance, mode, bg_image, bg_image_pil, bg_image_pil_no_background, image_path,\
        image_no_background_path, scale_json_file
    st.header("Image")
    st.markdown(
        """
        Upload Disk Image, preprocess, set scale unit and remove background
        """
    )

    selected = option_menu(None, [Menu.image, Menu.scale, Menu.background],
         icons=['house', 'gear'], menu_icon="cast", default_index=0, orientation="horizontal")


    if selected == Menu.image:
        mode =  st.radio( "Mode",("cross-section", "core"), horizontal=True )

        bg_image = st.file_uploader("Image:", type=["png", "jpg"]) if bg_image is  None else bg_image

        if bg_image:
            bg_image_pil = Image.open(bg_image)
            bg_image_pil.save(image_path)
            #resize image
            bg_image_pil_display = bg_image_pil.resize((256, 256), Image.Resampling.LANCZOS)
            st.image(bg_image_pil_display)


    if selected == Menu.scale and bg_image:
        units_mode = st.radio(
            "Unit:",
            ("cm", "mm", "dpi"), horizontal=True
        )
        if units_mode == "dpi":
            scale_pixels = st.number_input("DPI scale:", 1, 2000, 300)

        else:
            st.button("Set Distance in Pixels", on_click=set_scale, kwargs={"image_path":image_path,
                                                                            "output_file":scale_json_file})
            st.number_input("Distance in Pixels", 1, 10000, pixels_length)
            st.number_input("Know distance", 1, 100, know_distance)

    #st.subheader("Remove Background")
    if selected == Menu.background and bg_image:
        if st.button("Remove Background"):
            interface = BackgroundInterface(image_path, background_json_path, image_no_background_path)
            interface.interface()
            res = interface.parse_output()
            if res is not  None:
                bg_image_pil_no_background = interface.remove_background()
                bg_image_pil_no_background = bg_image_pil_no_background.resize((DISPLAY_IMAGE_SIZE, DISPLAY_IMAGE_SIZE),
                                                                               Image.Resampling.LANCZOS)

        if bg_image_pil_no_background:
            st.image(bg_image_pil_no_background)




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


def set_scale(image_path, output_file):
    global pixels_length
    scale = ScaleInterface(image_path, output_file)
    scale.interface()
    pixels_length = scale.parse_output()

    return pixels_length