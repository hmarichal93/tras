import streamlit as st
from streamlit_option_menu import option_menu

from ui.common import Context, RunningWidget
from ui.batch.metadata import main as metadata

class ViewContext(Context):
    def init_specific_ui_components(self):
        config = self.config["image"]

        self.image_path = self.output_dir / config["image_path"]
        self.mode = config["mode"]

        self.scale_json_path = self.output_dir / config["scale"]["json_path"]
        self.units_mode = config["scale"]["unit"]
        self.pixels_length = config["scale"]["pixels_length"]
        self.know_distance = config["scale"]["know_distance"]
        self.dpi = config["scale"]["dpi"]
        self.scale_status = config["scale"]["status"]


        self.tree_planting_date = config["metadata"]["tree_planting_date"]
        self.harvest_date = config["metadata"]["harvest_date"]
        self.location = config["metadata"]["location"]
        self.species = config["metadata"]["species"]
        self.observations = config["metadata"]["observations"]
        self.code = config["metadata"]["code"]
        self.latitude = config['metadata']['latitude']
        self.longitude = config['metadata']['longitude']

        self.image_no_background_path = self.output_dir / config["background"]["image_path"]
        self.background_json_path = self.output_dir / config["background"]["json_path"]
        self.resize_factor = config["background"]["resize_factor"]

        #runtime variables
        self.bg_image = None
        self.bg_image_pil = None
        self.bg_image_pil_no_background = None


    def update_config(self):
        config = self.config["image"]

        config["mode"] = self.mode
        config["image_path"] = str(self.image_path.name)


        config["scale"]["unit"] = self.units_mode
        config["scale"]["pixels_length"] = self.pixels_length
        config["scale"]["know_distance"] = self.know_distance
        config["scale"]["dpi"] = self.dpi
        config["scale"]["status"] = self.scale_status


        config["metadata"]["tree_planting_date"] = self.tree_planting_date
        config["metadata"]["harvest_date"] = self.harvest_date
        config["metadata"]["location"] = self.location
        config["metadata"]['latitude'] = self.latitude
        config['metadata']['longitude'] = self.longitude
        config["metadata"]["species"] = self.species
        config["metadata"]["observations"] = self.observations
        config["metadata"]["code"] = self.code

        config["background"]["resize_factor"] = self.resize_factor

        return

        # config["background"]["image_path"] = self.image_no_background_path
        # config["background"]["json_path"] = self.background_json_path



class Menu:
    metadata = "Metadata"
    configuration = "Configuration"
    pith = "Pith"
    background = "Background"
    automatic = "Ring Delineation"
    manual = "Ring Editing"
    metrics = "Metrics"
    export = "Export"

def main(runtime_config_path):
    CTX = ViewContext(runtime_config_path)
    CTX.init_specific_ui_components()

    st.header("Batch Processing")
    st.markdown(
        """
        """
    )

    selected = option_menu(None, [Menu.metadata, Menu.configuration, Menu.pith, Menu.background,
                                  Menu.automatic, Menu.manual, Menu.metrics, Menu.export],
                           menu_icon="cast", default_index=0, orientation="horizontal")
    if selected == Menu.metadata:
        metadata(runtime_config_path)



    return