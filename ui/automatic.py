import streamlit as st

from streamlit_option_menu import option_menu
from lib.image import LabelMeInterface as UserInterface

class Menu:
    pith = "Pith"
    cstrd = "CS-TRD"
    inbd = "INBD"

def annotate_pith(input_image=None, output_file=None):


    return
def main(bg_image_pil, pith_path):
    selected = option_menu(None, [Menu.pith, Menu.cstrd, Menu.inbd],
        icons=['house', 'gear'], menu_icon="cast", default_index=0, orientation="horizontal")

    if selected == Menu.pith:
        pith_button = st.button("Pith", on_click=annotate_pith, kwargs = {"input_image":bg_image_pil,
                                                                          "output_file": pith_path})