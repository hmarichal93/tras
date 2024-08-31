import streamlit as st
import os

from streamlit_option_menu import option_menu
from pathlib import Path

from lib.io import load_json, write_json, bytesio_to_dict



def main(default_config_path, runtime_config_path):

    uploaded_config = st.file_uploader("Runtime Config", type=["json"])

    if not uploaded_config:
        display_runtime_config_path = "default"
        config = load_json(default_config_path)

    else:
        display_runtime_config_path = uploaded_config.name
        config = bytesio_to_dict(uploaded_config)

    st.write('Runtime Config `%s`' % display_runtime_config_path)
    if not Path(runtime_config_path).exists():
        write_json(config, runtime_config_path)

    reset = st.button("Clearing Previous Experiments")
    if reset:
        delete_cache_folder(config["general"]["output_dir"])
        reset_runtime_config(runtime_config_path, default_config_path)

    return runtime_config_path

def reset_runtime_config(runtime_config_path, default_config_path):
    config = load_json(default_config_path)
    write_json(config, runtime_config_path)
    st.write("Runtime Config reset to default")

def delete_cache_folder(folder):
    if os.path.exists(folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
        st.write("Cache folder `%s` cleared" % folder)
    else:
        st.write("Cache folder `%s` does not exist" % folder)