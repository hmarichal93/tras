from abc import ABC, abstractmethod
from pathlib import Path
import streamlit as st

from lib.io import load_json, write_json, read_file_as_binary, bytesio_to_dict


class Shapes:
    pith = "Pith"
    latewood = "Late Wood"
    earlywood = "Early Wood"
    knot = "Knot"
    compresionwood = "Compression Wood"

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

def file_uploader(label, output_file, extension):
    uploaded_cw_annotation_file = st.file_uploader(label, type=[extension])
    if uploaded_cw_annotation_file:
        save_annotation_file_locally(output_file, uploaded_cw_annotation_file)

    return output_file