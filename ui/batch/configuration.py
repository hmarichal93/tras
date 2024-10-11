import streamlit as st

from ui.common import Context, RunningWidget, set_date_input, select_directory


class ViewContext(Context):
    def init_specific_ui_components(self):
        config = self.config["image"]
        self.tree_planting_date = config["metadata"]["tree_planting_date"]
        self.harvest_date = config["metadata"]["harvest_date"]
        self.location = config["metadata"]["location"]
        self.species = config["metadata"]["species"]
        self.observations = config["metadata"]["observations"]
        self.code = config["metadata"]["code"]
        self.latitude = config['metadata']['latitude']
        self.longitude = config['metadata']['longitude']





    def update_config(self):
        config = self.config["image"]

        config["metadata"]["tree_planting_date"] = self.tree_planting_date
        config["metadata"]["harvest_date"] = self.harvest_date
        config["metadata"]["location"] = self.location
        config["metadata"]['latitude'] = self.latitude
        config['metadata']['longitude'] = self.longitude
        config["metadata"]["species"] = self.species
        config["metadata"]["observations"] = self.observations
        config["metadata"]["code"] = self.code




        return





class UI:

    def __init__(self, runtime_config_path):
        CTX = ViewContext(runtime_config_path)
        CTX.init_specific_ui_components()
        self.CTX = CTX





def main(runtime_config_path):
    ui = UI(runtime_config_path)

    st.divider()


    return