import pandas as pd
import streamlit as st

from ui.common import Context, RunningWidget, set_date_input, select_directory, file_uploader


class Scale:
    nanometer = "nm"
    micrometer = "micrometer"
    millimeter = "mm"
    centimeter = "cm"
    dpi = "dpi"
    general = 'General'
    per_sample = 'Per Sample'
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

        self.scale_json_path = self.output_dir / config["scale"]["json_path"]
        self.units_mode = config["scale"]["unit"]
        self.pixels_length = config["scale"]["pixels_length"]
        self.know_distance = config["scale"]["know_distance"]
        self.dpi = config["scale"]["dpi"]
        self.scale_status = config["scale"]["status"]



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

        config["scale"]["unit"] = self.units_mode
        config["scale"]["pixels_length"] = self.pixels_length
        config["scale"]["know_distance"] = self.know_distance
        config["scale"]["dpi"] = self.dpi
        config["scale"]["status"] = self.scale_status

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
        display = st.checkbox("Site Data", value=True)
        if not display:
            return

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

    def check_dataframe(self, df):
        #check size
        if df.shape[0] == 0:
            st.error("The csv file is empty")
            return False

        columns = ['index', 'sample', 'pixels_length', 'know_distance', 'unit']
        if not list(df.columns) == columns:
            st.error(f"Columns in the csv file must be {columns}")
            return False

        # check that the unit column has the correct values
        units = [Scale.nanometer, Scale.micrometer, Scale.millimeter, Scale.centimeter, Scale.dpi]
        if not df['unit'].isin(units).all():
            st.error(f"Unit column must have the values {units}")
            return False

        # check that the know_distance column is a number
        if not pd.to_numeric(df['know_distance'], errors='coerce').notnull().all():
            st.error("know_distance column must be a number")
            return False

        # check that the pixels_length column is a number
        if not pd.to_numeric(df['pixels_length'], errors='coerce').notnull().all():
            st.error("pixels_length column must be a number")
            return False

        #check that the sample column is not empty
        if df['sample'].isnull().any():
            st.error("sample column must not be empty")
            return False

        #check that the sample column is unique
        if not df['sample'].is_unique:
            st.error("sample column must be unique")
            return False

        #check that the index column is unique
        if not df.index.is_unique:
            st.error("index column must be unique")
            return False

        df.set_index('index', inplace=True)

        images = [file for file in self.CTX.output_dir.glob("*") if file.suffix in
                  ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp']]
        if not set(df['sample']).issubset(set([image.stem for image in images])):
            st.error("Some images in the output directory do not have a corresponding row in the csv file")
            return

        return True

    def scale_settings(self):
        display = st.checkbox("Scale Settings", value=True)
        if not display:
            return

        col1, col2 = st.columns([1, 3])
        with col1:
            self.CTX.scale_status = st.selectbox("Scale mode", [Scale.general, Scale.per_sample], index=0,
                                                 help="General: The scale is the same for all the samples\n"
                                                      ". Per Sample: The scale is different for each sample")

            if self.CTX.scale_status == Scale.per_sample:
                # download template for csv file using dataframe
                df = pd.DataFrame(columns=['index', 'sample', 'pixels_length', 'know_distance', 'unit'])
                df.set_index('index', inplace=True)
                @st.cache_data
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode("utf-8")
                csv = convert_df(df)
                st.download_button(
                    label="Download CSV Template",
                    data=csv,
                    file_name="template.csv",
                    mime="text/csv",
                )

        with col2:
            if self.CTX.scale_status == Scale.general:
                self.CTX.units_mode = st.selectbox("Units", [Scale.nanometer, Scale.micrometer, Scale.millimeter,
                                                             Scale.centimeter, Scale.dpi], index=0)
                if self.CTX.units_mode == Scale.dpi:
                    self.CTX.dpi = st.number_input("DPI", value=self.CTX.dpi)

                else:
                    self.CTX.pixels_length = st.number_input("Length in pixels", value=self.CTX.pixels_length)
                    self.CTX.know_distance = st.number_input("Known distance", value=self.CTX.know_distance)

            if self.CTX.scale_status == Scale.per_sample:
                file_uploader('Upload scale csv file', self.CTX.scale_json_path, '.csv',
                              help_text="The csv file must have the columns: 'sample', 'pixels_length', 'know_distance',"
                                        " 'unit'. Where 'sample' is the name of the sample (no extension),"
                                        " 'pixels_length' is the length of the scale bar in pixels, 'know_distance' "
                                        "is the distance to which refers the 'pixels_length' column  and 'unit' is the unit of the length. "
                                        "If 'unit' is dpi, the 'know_distance' column is not used (empty) and 'pixels_length' is the dpi value. "
                                        "'unit' can be 'nm', 'micrometer', 'mm', 'cm' or 'dpi'")
                if self.CTX.scale_json_path.exists():
                    df = pd.read_csv(self.CTX.scale_json_path)
                    if self.check_dataframe(df):

                        st.write(df)





        return
def main(runtime_config_path):
    ui = UI(runtime_config_path)

    st.divider()
    ui.image_set_location()

    st.divider()
    ui.form()

    st.divider()
    ui.scale_settings()

    st.divider()
    ui.CTX.save_config()
    return
