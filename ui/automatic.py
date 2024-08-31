import streamlit as st

from streamlit_option_menu import option_menu
from pathlib import Path

from lib.image import LabelMeInterface as UserInterface
from ui.common import Context

class Menu:
    pith = "Pith"
    cstrd = "CS-TRD"
    inbd = "INBD"

def annotate_pith():


    return


class ViewContext(Context):
    def init_specific_ui_components(self):
        config = self.config["image"]

        self.image_path = self.output_dir / config["image_path"]
        self.image_no_background_path = self.output_dir / config["background"]["image_path"]
        if Path(self.image_no_background_path).exists():
            self.image_path = self.image_no_background_path


        return

    def update_config(self):


        return


def main(runtime_config_path):
    global CTX
    CTX = ViewContext(runtime_config_path)
    CTX.init_specific_ui_components()

    selected = option_menu(None, [Menu.pith, Menu.cstrd, Menu.inbd],
                           menu_icon="cast", default_index=0, orientation="horizontal")

    if selected == Menu.pith:
        pith_button = st.button("Pith")
        #if pith_button:




    #save status
    CTX.save_config()