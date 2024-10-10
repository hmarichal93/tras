import os

import streamlit as st

from streamlit_option_menu import option_menu
from PIL import Image
from pathlib import Path

from ui.image import main as image
from ui.automatic import main as automatic_ring_delineation
from ui.home import main as home
from ui.manual import  main as manual
from ui.metrics import  main as metrics
from ui.save import main as save
from ui.update import pull_last_changes_from_remote_repo
from ui.batch.batch import main as batch

from lib.io import load_json, write_json, bytesio_to_dict


APP_NAME = "DendroTool: An Interactive Software for tracing Tree Ring Cross Sections"
DEFAULT_CONFIG_PATH = "./config/default.json"
RUNTIME_CONFIG_PATH = "./config/runtime.json"


class Menu:
    home = "Home"
    image = "Image"
    automatic_ring_delineation = "Automatic Ring Delineation"
    manual_ring_delineation = "Manual Ring Delineation"
    metrics = "Metrics"
    save = "Save"





def initialization():
    if Path(RUNTIME_CONFIG_PATH).exists():
        os.system(f"rm -rf {RUNTIME_CONFIG_PATH}")

    return

class Mode:
    single = "Single"
    batch = "Batch"

def main():
    im = Image.open('assets/pixels_wood.jpg')
    # Adding Image to web app
    st.set_page_config(page_title=APP_NAME, page_icon=im, layout='wide')
    st.markdown(
        """
        <style>
        /* Change the width of the sidebar */
        [data-testid="stSidebar"] {
            width: 33.33% !important;
        }
        /* Adjust the main content to fit within the remaining width */
        [data-testid="stSidebar"] + div {
            width: 66.67%;
            margin-left: 33.33%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title(APP_NAME)

    # 1. as sidebar menu
    with st.sidebar:
        st.markdown(
            """
            <a href="https://github.com/hmarichal93/dendrotool" target="_blank">
                <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="30" height="30">
            </a>
            """,
            unsafe_allow_html=True
        )
        mode = st.radio("Mode", [Mode.single, Mode.batch], index=0, horizontal=True, help="Select the mode to run the app. Single) one image, Batch) batch of images.")
        if mode == Mode.single:
            selected = option_menu("", [Menu.home, Menu.image, Menu.automatic_ring_delineation,
                                              Menu.manual_ring_delineation, Menu.metrics, Menu.save], menu_icon="cast", default_index=0)


    if mode == Mode.single:
        if selected == Menu.home:
            home(DEFAULT_CONFIG_PATH, RUNTIME_CONFIG_PATH)

        elif selected == Menu.image:
            image(RUNTIME_CONFIG_PATH)

        elif selected == Menu.automatic_ring_delineation:
            automatic_ring_delineation(RUNTIME_CONFIG_PATH)

        elif selected == Menu.manual_ring_delineation:
            manual(RUNTIME_CONFIG_PATH)

        elif selected == Menu.metrics:
            metrics(RUNTIME_CONFIG_PATH)

        elif selected == Menu.save:
            save(RUNTIME_CONFIG_PATH)


    if mode == Mode.batch:
        batch(DEFAULT_CONFIG_PATH)

    with st.sidebar:

        if st.button("Update", help="Update the app to the latest version"):
            pull_last_changes_from_remote_repo(os.path.dirname(os.path.abspath(__file__)))

        st.image("assets/wood_image.jpeg")




if __name__ == "__main__":
    main()