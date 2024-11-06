import streamlit as st
import os

from streamlit_option_menu import option_menu
from pathlib import Path

from lib.io import load_json, write_json, bytesio_to_dict
from ui.common import Context


class ViewContext(Context):
    def init_specific_ui_components(self):
        config = self.config["image"]
        self.image_path = self.output_dir / config["image_path"]


    def update_config(self):
        self.config["general"]["output_dir"] = str(self.output_dir)



class UI:

    def __init__(self, runtime_config_path):
        st.header("Home")
        st.markdown(
            """
            Clear previous experiments, upload a runtime config, or select an output folder.
            """
        )
        st.divider()

    def upload_config(self, default_config_path, runtime_config_path):
        self.default_config_path = default_config_path
        self.runtime_config_path = runtime_config_path
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

        CTX = ViewContext(runtime_config_path)
        CTX.init_specific_ui_components()
        self.CTX = CTX

        return

    def clear_experiments(self):
        reset = st.button("Clear Previous Experiments")
        if reset:
            delete_cache_folder(self.CTX.output_dir)
            script_path = os.path.abspath(__file__)
            root = Path(script_path).parent.parent
            static_files_dir = Path(root) / "static"
            delete_cache_folder(str(static_files_dir))
            metric_dir = Path(self.CTX.output_dir) / "metrics"
            delete_cache_folder(str(metric_dir))
            inbd_dir = Path(self.CTX.output_dir) / "inbd"
            os.system(f"rm -rf {inbd_dir}")
            delete_cache_folder(str(inbd_dir))
            inbd_dir = Path(self.CTX.output_dir) / "cstrd"
            os.system(f"rm -rf {inbd_dir}")
            delete_cache_folder(str(inbd_dir))
            pith_dir = Path(self.CTX.output_dir) / "pith_mask"
            delete_cache_folder(str(pith_dir))
            # delete streamlit cache
            st.cache_data.clear()
            # delete tmp folder
            tmp_dir = Path(self.CTX.output_dir) / "tmp"
            delete_cache_folder(str(tmp_dir))

            reset_runtime_config(self.runtime_config_path, self.default_config_path)
            st.rerun()


    def folder_picker(self):
        folder_picker = st.button("Select Output folder")
        dirname = False
        if folder_picker:
            import tkinter as tk
            from tkinter import filedialog

            # Set up tkinter
            root = tk.Tk()
            root.withdraw()

            # Make folder picker dialog appear on top of other windows
            root.wm_attributes('-topmost', 1)
            dirname = filedialog.askdirectory(master=root)
            st.write(f'Selected folder: {dirname}')
            dirname = dirname if Path(dirname).is_dir() else False
            if dirname:
                self.CTX.output_dir = dirname

        return self.runtime_config_path




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

def main(default_config_path, runtime_config_path):
    ui = UI(default_config_path)

    col1, col2 = st.columns([0.6, 1])
    with col1:
        ui.upload_config(default_config_path, runtime_config_path)

    col1, col2, col3 = st.columns([0.3, 0.3, 1])

    with col2:
        ui.clear_experiments()

    with col1:
        ui.folder_picker()

    ui.CTX.save_config()

    return runtime_config_path