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

    def image_set_location(self):
        dirname = select_directory("Imageset directory")
        if dirname:
            self.CTX.output_dir = dirname

    def form(self):
        code = st.text_input("Code", value=self.CTX.code)
        self.CTX.code = code

        col1, col2 = st.columns([1, 3])
        with col1:
            checkbox_harvest = st.checkbox("Harvest date", value=False,
                                           help='Check if the date is the harvest date or the planting date.'
                                                ' It is used to set the correct date for each ring')
        with col2:
            if not checkbox_harvest:
                self.CTX.tree_planting_date = set_date_input(self.CTX.tree_planting_date, "Tree planting date")
                self.CTX.harvest_date = None
            else:
                self.CTX.harvest_date = set_date_input(self.CTX.harvest_date, "Harvest date")
                self.CTX.tree_planting_date = None


        location = st.text_input("Location", value=self.CTX.location)
        self.CTX.location = location

        latitude = st.number_input("Latitude", value=self.CTX.latitude, format="%.8f")
        self.CTX.latitude = latitude

        longitude = st.number_input("Longitude", value=self.CTX.longitude, format="%.8f")
        self.CTX.longitude = longitude

        species = st.text_input("Species", value=self.CTX.species)
        self.CTX.species = species

        observations = st.text_area("Observations", value=self.CTX.observations)
        self.CTX.observations = observations

        return
def main(runtime_config_path):
    ui = UI(runtime_config_path)

    st.divider()
    ui.image_set_location()

    st.divider()
    ui.form()

    ui.CTX.save_config()

    return
