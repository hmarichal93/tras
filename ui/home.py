import streamlit as st
import os

from streamlit_option_menu import option_menu

def main():
    filename = st.file_uploader("Runtime Config", type=["json"])
    return filename