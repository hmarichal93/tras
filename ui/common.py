from abc import ABC, abstractmethod
from pathlib import Path
import streamlit as st
from streamlit_image_zoom import image_zoom
import altair as alt
import cv2

from lib.io import load_json, write_json, read_file_as_binary, bytesio_to_dict
from lib.image import  load_image, resize_image_using_pil_lib
from backend.labelme_layer import ring_relabelling


class Pith:
    pixel = "Pixel"
    boundary = "Boundary"
    manual = "Manual"
    automatic = "Automatic"
    apd = "APD"
    apd_pl = "APD-PL"
    apd_dl = "APD-DL"

class Shapes:
    pith = "Pith"
    latewood = "Late Wood"
    earlywood = "Early Wood"
    other = "Other"

class Color:
    red = "red"
    green = "green"
    blue = "blue"
    yellow = "yellow"
    black = "black"
    white = "white"

class Context(ABC):
    def __init__(self, runtime_config_path):
        #set general config
        config = load_json(runtime_config_path)
        self.output_dir = Path(config["general"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_config_path = runtime_config_path

        self.display_image_size = config["general"]["display_image_size"]

        self.config = config


    def save_config(self):
        self.update_config()
        write_json(self.config, self.runtime_config_path)

    @abstractmethod
    def init_specific_ui_components(self):
        pass


    @abstractmethod
    def update_config(self):
        pass


def download_button(file_path: str, label: str, filen_name:str, mime:str)-> None:
    """
    Streamlit binary download button
    :param file_path: file path to read
    :param label: label to display in the button
    :param filen_name: name of the file to download
    :param mime:
    :return:
    """
    if not Path(file_path).exists():
        st.error(f"File {file_path} does not exist")
        return
    file_content = read_file_as_binary(file_path)
    #2.2 download the zip file
    st.download_button(label=label, data=file_content, file_name=filen_name, mime=mime)

    return



def save_annotation_file_locally(filename, file_uploader_instance):
    config = bytesio_to_dict(file_uploader_instance)
    write_json(config, filename)

def file_uploader(label, output_file, extension, CTX = None):
    uploaded_cw_annotation_file = st.file_uploader(label, type=[extension])
    if uploaded_cw_annotation_file:
        save_annotation_file_locally(output_file, uploaded_cw_annotation_file)
        if CTX is not None:
            # relabel rings
            if CTX.autocomplete_ring_date:
                ring_relabelling(CTX.image_path, output_file, CTX.harvest_date)

    return output_file


def display_image(image_path, width: int = 320):
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image(image_path, width=width)

def display_chart(chart):
    col1, col2, col3 = st.columns([0.5, 2, 0.5])
    with col2:
        st.altair_chart(chart)


def plot_chart(df, title=""):
    x, y = df.columns

    line_chart = alt.Chart(df).mark_line(color="#FF5733").encode(
        x=alt.X(x,sort=None),
        y=y
    )

    points = alt.Chart(df).mark_point(color="#FF5733", filled=True, size=100).encode(
        x=alt.X(x,sort=None),
        y=y
    )

    chart = (line_chart + points).properties(
        width=800,
        height=400,
        title=alt.TitleParams(title, anchor='middle', offset=20)
    )

    display_chart(chart)

class RunningWidget:
    def __init__(self):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            gif_runner = st.empty()

            with gif_runner.container():
                st.markdown('<div class="centered"><img src="./assets/rotating_wood.gif" width="300"></div>',
                            unsafe_allow_html=True)

            gif_runner.image("./assets/rotating_wood.gif")

        self.gif_runner = gif_runner

    def empty(self):
        self.gif_runner.empty()

def display_image_with_zoom(image_path):
    image = load_image(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = 800, 400
    image_z = resize_image_using_pil_lib(image, height, width)
    height, width = image_z.shape[:2]
    col1, col2, col3 = st.columns([0.5, 2, 0.5])
    with col2:
        image_zoom(image_z, mode="scroll", size=(width, height), keep_aspect_ratio=True, zoom_factor=4.0,
               increment=0.2)

    return

def display_data_editor(data):

    st.data_editor(data,
                   column_config= {
                       'image': st.column_config.ImageColumn('Preview Ring', help="Preview Ring")
                   },
                   hide_index = True
    )

class Pith:
    pixel = "Pixel"
    boundary = "Boundary"
    manual = "Manual"
    automatic = "Automatic"
    apd = "APD"
    apd_pl = "APD-PL"
    apd_dl = "APD-DL"


def check_image(CTX):
    if not Path(CTX.image_path).exists():
        st.error("Image not found. Please upload an image in the *Upload Image* view")
        return True
    return False