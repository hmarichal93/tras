from abc import ABC, abstractmethod
from pathlib import Path
import streamlit as st
import datetime

from lib.io import load_json, write_json, read_file_as_binary, bytesio_to_dict


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



def save_annotation_file_locally(filename, file_uploader_instance, extension=".json"):
    if extension == ".json":
        config = bytesio_to_dict(file_uploader_instance)
        write_json(config, filename)

    if extension == ".csv":
        # save the file as csv
        with open(filename, "wb") as f:
            f.write(file_uploader_instance.getvalue())



def file_uploader(label, output_file, extension='.json', help_text=None):
    uploaded_cw_annotation_file = st.file_uploader(label, type=[extension], help=help_text)
    if uploaded_cw_annotation_file:
        save_annotation_file_locally(output_file, uploaded_cw_annotation_file, extension)

    return output_file


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


def set_date_input(dictionary_date, text="Tree planting date"):
    if dictionary_date is None:
        # default value
        dictionary_date = {'year': 2000, 'month': 1, 'day': 1}

    year, month, day = (dictionary_date['year'], dictionary_date['month'],
                        dictionary_date['day'])

    input_date = st.date_input(text, datetime.date(year, month, day),
                               min_value=datetime.date(1500, 1, 1),
                               max_value=datetime.date(3000, 1, 1))
    dictionary_date['year'] = input_date.year
    dictionary_date['month'] = input_date.month
    dictionary_date['day'] = input_date.day
    return dictionary_date


def select_directory(text="Select Output folder"):
    folder_picker = st.button(text)
    dirname = False
    if folder_picker:
        import tkinter as tk
        from tkinter import filedialog

        # Set up tkinter
        root = tk.Tk()
        root.withdraw()

        # Make folder picker dialog appear on top of other windows
        root.wm_attributes('-topmost', 1)
        dirname = filedialog.askdirectory( master = root)
        st.write(f'Selected folder: {dirname}')
        dirname = dirname if Path(dirname).is_dir() else False

    return dirname