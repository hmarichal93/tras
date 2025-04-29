import streamlit as st
import numpy as np
import cv2
import datetime
import os

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from streamlit_option_menu import option_menu
from pathlib import Path

from lib.image import  write_image, resize_image
from ui.common import Context, RunningWidget, Pith, display_image, check_image, resize_slider
from backend.labelme_layer import (LabelmeShapeType,
                                   LabelmeObject, LabelmeInterface as UserInterface, resize_annotations,
                                   AL_LateWood_EarlyWood)
from urudendro.remove_salient_object import remove_salient_object

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
        self.pixel_per_mm = config["scale"]["pixel_per_mm"]


        #self.tree_planting_date = config["metadata"]["tree_planting_date"]
        self.harvest_date = config["metadata"]["harvest_date"]
        self.location = config["metadata"]["location"]
        self.species = config["metadata"]["species"]
        self.observations = config["metadata"]["observations"]
        self.code = config["metadata"]["code"]
        self.latitude = config['metadata']['latitude']
        self.longitude = config['metadata']['longitude']
        self.autocomplete_ring_date = config['metadata']['autocomplete_ring_date']

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
        config["scale"]["pixel_per_mm"] = self.pixel_per_mm


        #config["metadata"]["tree_planting_date"] = self.tree_planting_date
        config["metadata"]["harvest_date"] = self.harvest_date
        config["metadata"]["location"] = self.location
        config["metadata"]['latitude'] = self.latitude
        config['metadata']['longitude'] = self.longitude
        config["metadata"]["species"] = self.species
        config["metadata"]["observations"] = self.observations
        config["metadata"]["code"] = self.code
        config['metadata']['autocomplete_ring_date'] = self.autocomplete_ring_date

        config["background"]["resize_factor"] = self.resize_factor

        return



class Menu:
    image = "Upload Image"
    scale = "Set Scale"
    preprocess  = "Preprocess"
    metadata = "Metadata"



def set_date_input(dictionary_date, text="Harvest date", help=""):
    default_year = int(dictionary_date['year'])
    current_year = datetime.datetime.now().year
    min_year = current_year - 200
    max_year = current_year
    years_list = list(reversed(range(min_year, max_year + 1)))
    selected_year = st.selectbox(text, years_list , index=years_list.index(default_year), help=help)

    dictionary_date['year'] = selected_year
    dictionary_date['month'] = 1
    dictionary_date['day'] = 1
    return dictionary_date






class UI:

    def __init__(self, runtime_config_path):
        st.header("Image")
        st.markdown(
            """
            Upload Disk Image, preprocess, set scale unit and add metadata. We suggest to resize images with dimensions
            higher than 1500px.
            """
        )
        st.divider()
        CTX = ViewContext(runtime_config_path)
        CTX.init_specific_ui_components()
        self.CTX = CTX

    def option_menu(self):
        selected = option_menu(None, [Menu.image, Menu.preprocess, Menu.scale, Menu.metadata],
                               menu_icon="cast", default_index=0, orientation="horizontal")
        return selected


    def upload_image(self):
        self.CTX.bg_image = st.file_uploader("Image:", type=["png", "jpg"])
        gif_runner = RunningWidget()
        if self.CTX.bg_image is not None:
            #get code
            image_name = self.CTX.bg_image.name
            if image_name.endswith(".png"):
                self.CTX.code = image_name.replace(".png", "")
            elif image_name.endswith(".jpg"):
                self.CTX.code = image_name.replace(".jpg", "")
            os.system(f"rm -rf {self.CTX.output_dir}/*.png")
            os.system(f"rm -rf {self.CTX.output_dir}/*.json")
            os.system(f"rm -rf {self.CTX.output_dir}/metrics")
            Path(self.CTX.output_dir).mkdir(parents=True, exist_ok=True)
            self.CTX.scale_status = False

            self.CTX.bg_image_pil = Image.open(self.CTX.bg_image)
            self.CTX.bg_image_pil.save(self.CTX.image_path)
            bg_image_pil_display = self.CTX.bg_image_pil.resize((self.CTX.display_image_size,
                                                                 self.CTX.display_image_size),
                                                           Image.Resampling.LANCZOS)

            height, width = self.CTX.bg_image_pil.size
            st.write(f"Image dimensions: (H,W) = ({height}, {width})")
            display_image(bg_image_pil_display)

        elif Path(self.CTX.image_path).exists():
            self.CTX.bg_image_pil = Image.open(self.CTX.image_path)
            bg_image_pil_display = self.CTX.bg_image_pil.resize((self.CTX.display_image_size,
                                                                 self.CTX.display_image_size),
                                                           Image.Resampling.LANCZOS)
            #st.image(bg_image_pil_display)
            display_image(bg_image_pil_display)
        gif_runner.empty()

    def remove_background(self, radio):
        interface = BackgroundInterface(self.CTX.image_path, self.CTX.background_json_path,
                                        self.CTX.image_no_background_path)
        if radio == Pith.manual:
            interface.interface()
        else:
            interface.automatic()
        res = interface.parse_output()
        if res is not None:
            self.CTX.bg_image_pil_no_background = interface.remove_background()
            self.CTX.bg_image_pil_no_background = self.CTX.bg_image_pil_no_background.resize(
                (self.CTX.display_image_size, self.CTX.display_image_size), Image.Resampling.LANCZOS)

        return

    def resize_image(self, resize_factor):
        _, height, width = resize_image(self.CTX.image_path, resize_factor)

        if Path(self.CTX.image_no_background_path).exists():
            backup_image_path = str(self.CTX.image_no_background_path).replace(".png", "_bkp.png")
            os.system(f"cp {self.CTX.image_no_background_path} {backup_image_path}")
            self.CTX.image_no_background_path, _, _ = resize_image(self.CTX.image_no_background_path, resize_factor)
            resize_image(str(self.CTX.image_no_background_path).replace(".png", "_mask.png"),
                         resize_factor)
            new_annotations_path = resize_annotations(backup_image_path, self.CTX.image_no_background_path,
                                                      self.CTX.background_json_path)
            os.system(f"cp {new_annotations_path} {self.CTX.background_json_path}")

        self.CTX.scale_status = False
        return height, width

    def preprocess(self):
        if check_image(self.CTX):
            return None

        st.write(" ")
        st.write("Crop the image to remove the background. Draw a rectangle around the disk.")

        if st.button("Crop", help="Crop the image to remove the background. Draw a rectangle around the disk."):
            gif_runner = RunningWidget()
            crop_image(self.CTX)
            gif_runner.empty()

        #space
        st.write(" ")
        st.write(" ")
        radio = st.radio("Remove Background", (Pith.manual, Pith.automatic), index=0, horizontal=True)

        if st.button("Remove Background"):
            gif_runner = RunningWidget()
            self.remove_background(radio)
            gif_runner.empty()

        if Path(self.CTX.image_no_background_path).exists():
            self.CTX.bg_image_pil_no_background = Image.open(self.CTX.image_no_background_path)
            self.CTX.bg_image_pil_no_background = self.CTX.bg_image_pil_no_background.resize(
                (self.CTX.display_image_size, self.CTX.display_image_size), Image.Resampling.LANCZOS)
            display_image(self.CTX.bg_image_pil_no_background)

        resize_factor = resize_slider(default=self.CTX.resize_factor)
        if resize_factor != self.CTX.resize_factor and resize_factor > 0:
            self.CTX.resize_factor = resize_factor

        if st.button("Resize Image"):
            gif_runner = RunningWidget()
            height, width = self.resize_image(resize_factor)
            gif_runner.empty()
            st.warning(f"Image resized. Please set the scale again. Image new dimensions are:"
                       f" (H,W)=( {int(height)}, {int(width)}). Ring detection method works with dimentions lower"
                       f" than 1500px")

        return

    def set_scale(self):
        if check_image(self.CTX):
            return None

        self.CTX.units_mode = st.radio(
            "Unit:",
            ("nm", r"$\mu$m", "mm", "cm", "dpi"), horizontal=True, index=scale_index_unit(self.CTX.units_mode)
        )
        if self.CTX.units_mode == "dpi":
            dpi = st.number_input("DPI scale:", 1, 2000, self.CTX.dpi)
            if self.CTX.dpi != dpi:
                self.CTX.scale_status = True
                self.CTX.dpi = dpi

        else:
            button = st.button("Set Distance in Pixels", help="Mark a line with the distance in pixels.")
            if button:
                gif_runner = RunningWidget()
                self.CTX.pixels_length = set_scale(self.CTX)
                self.CTX.scale_status = True
                gif_runner.empty()
            pixels_length = st.number_input("Distance in Pixels", 1.0, 10000.0, float(self.CTX.pixels_length))
            # input float number

            if pixels_length != self.CTX.pixels_length:
                self.CTX.scale_status = True
                self.CTX.pixels_length = pixels_length

            know_distance = st.number_input("Know distance", 1, 100, self.CTX.know_distance)
            if know_distance != self.CTX.know_distance:
                self.CTX.scale_status = True
                self.CTX.know_distance = know_distance

            scale = self.CTX.pixels_length / self.CTX.know_distance
            st.write(f"Scale: {scale:.2f} pixels/{self.CTX.units_mode}")
            self.CTX.pixel_per_mm = self.get_pixel_per_mm(self.CTX.pixels_length, self.CTX.know_distance,
                                                          self.CTX.units_mode)

        return
    @staticmethod
    def get_pixel_per_mm(pixel_length, know_distance, units_mode):
        if units_mode == "nm":
            return pixel_length / know_distance * 10 ** -6
        elif units_mode == r"$\mu$m":
            return pixel_length / know_distance * 10 ** -3
        elif units_mode == "mm":
            return pixel_length / know_distance
        elif units_mode == "cm":
            return pixel_length / know_distance * 10
        elif units_mode == "dpi":
            return pixel_length / know_distance
    def metadata(self):
        if check_image(self.CTX):
            return None

        col1, col2 = st.columns([1, 1])
        with col1:
            code = st.text_input("Code", value=self.CTX.code)
            self.CTX.code = code

            self.CTX.harvest_date = set_date_input(self.CTX.harvest_date, "Harvest date")

            autocomplete_ring_date = st.checkbox("Autocomplete ring date", value=self.CTX.autocomplete_ring_date,
                                                 help="If checked, the ring date will be automatically filled with the harvest date. "
                                                      "This means that rings label will be created from the harvest date"
                                                      " (manually labeled rings will be deleted).")
            if autocomplete_ring_date != self.CTX.autocomplete_ring_date:
                self.CTX.autocomplete_ring_date = autocomplete_ring_date

            location = st.text_input("Location", value=self.CTX.location)
            self.CTX.location = location

            latitude = st.number_input("Latitude", value=self.CTX.latitude, format="%.8f")
            self.CTX.latitude = latitude

            longitude = st.number_input("Longitude", value=self.CTX.longitude, format="%.8f")
            self.CTX.longitude = longitude

        with col2:
            species = st.text_input("Species", value=self.CTX.species)
            self.CTX.species = species

            observations = st.text_area("Observations", value=self.CTX.observations)
            self.CTX.observations = observations


def main(runtime_config_path):
    ui = UI(runtime_config_path)

    selected = ui.option_menu()


    if selected == Menu.image:
        ui.upload_image()

    if selected == Menu.preprocess:
        ui.preprocess()

    if selected == Menu.scale:
        ui.set_scale()

    if selected == Menu.metadata:
        ui.metadata()

    #export status
    ui.CTX.save_config()

    return

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
        disk_mask = remove_salient_object(self.read_file_path, str(self.output_image_path))
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


def crop_image(CTX):
    #crop image
    crop = CropInterface(CTX.image_path, str(CTX.output_dir / "crop.json"))
    crop.interface()
    points = crop.parse_output()
    if points is not None:
        #crop image
        from urudendro.image import load_image, write_image
        ymin,xmin = points[0]
        ymax,xmax = points[1]
        image = load_image(CTX.image_path)
        write_image(CTX.image_path, image[ymin:ymax, xmin:xmax])
        bg_image_pil = Image.open(CTX.image_path)
        CTX.bg_image_pil_no_background = bg_image_pil.resize((CTX.display_image_size, CTX.display_image_size),
                                                             Image.Resampling.LANCZOS)
        display_image(CTX.bg_image_pil_no_background)

        st.write(f"Image cropped. New dimensions are: (H,W) = ({ymax-ymin}, {xmax-xmin})")

class CropInterface(UserInterface):

    def __init__(self, image_path, output_file):
        super().__init__(read_file_path = image_path, write_file_path=output_file)

    def parse_output(self):
        object = LabelmeObject(self.write_file_path)
        if len(object.shapes) > 1:
            st.error("More than one shape found. Add only one shape")
            return None
        shape = object.shapes[0]
        if not(shape.shape_type == LabelmeShapeType.rectangle):
            st.error("Shape is not a rectangle. Remember that you are cropping the image")
            return None

        return np.array(shape.points).astype(int)

    def from_structure_to_labelme_shape(self, structure_list):
        pass

    def from_labelme_shape_to_structure(self, shapes):
        pass