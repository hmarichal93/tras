import os
import streamlit as st

from pathlib import Path

from ui.common import Context, download_button, check_image





class VisualizationShape:
    def __init__(self, shapes_list, thickness, color, stroke, fill, opacity):
        self.shapes_list = shapes_list
        self.thickness = thickness
        self.color = color
        self.stroke = stroke
        self.fill = fill
        self.opacity = opacity

    def __str__(self):
        return f"({self.thickness=} {self.color=} {self.stroke=} {self.fill=} {self.opacity=})"

    def __repr__(self):
        return f"({self.thickness=} {self.color=} {self.stroke=} {self.fill=} {self.opacity=})"

class ViewContext(Context):
    def init_specific_ui_components(self):
        config = self.config["image"]
        self.image_path = self.output_dir / config["image_path"]
        self.code = config["metadata"]["code"]

        config = self.config["export"]
        self.path_csv = config["path_csv"]
        self.area_metrics_report = config["area_metrics_report"]
        self.image = config["image"]
        self.path_images = config["path_images"]
        self.ann_lw = config["ann_lw"]
        self.background = config["background"]
        self.path_pos = config["path_pos"]
        self.ann_ew = config["ann_ew"]
        self.area_csv = config["area_csv"]
        self.ann_other = config["ann_other"]


    def update_config(self):
        config = self.config["export"]
        config["path_csv"] = self.path_csv
        config["area_metrics_report"] = self.area_metrics_report
        config["image"] = self.image
        config["path_images"] = self.path_images
        config["ann_lw"] = self.ann_lw
        config["background"] = self.background
        config["path_pos"] = self.path_pos
        config["ann_ew"] = self.ann_ew
        config["area_csv"] = self.area_csv
        config["ann_other"] = self.ann_other






class UI:

    def __init__(self, runtime_config_path):
        st.header("Save")
        st.markdown(
            """
            This page allows you to download the results of the analysis.
            """
        )
        st.divider()
        CTX = ViewContext(runtime_config_path)
        CTX.init_specific_ui_components()
        self.CTX = CTX

    def generate_file_list_to_export(self):
        files_to_export = []
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            self.CTX.path_csv = st.checkbox("Path CSV", value = self.CTX.path_csv,
                                      help="Include the path csv files. Metrics>Path>CSV")

            self.CTX.area_metrics_report = st.checkbox("Area Metrics Report", value = self.CTX.area_metrics_report,
                                              help = "Include the area metrics report pdf. Metrics>Area")

            self.CTX.image = st.checkbox("Image", value = self.CTX.image,
                                    help = "Include the image file. Image")

        with col2:
            self.CTX.path_images = st.checkbox("Path Images", value= self.CTX.path_images,
                                      help="Include the path images files. Metrics>Path>Images")

            self.CTX.ann_lw = st.checkbox("Annotation Latewood", value=self.CTX.ann_lw,
                                 help = "Include the annotation latewood file. Ring Editing>Annotation")

            self.CTX.background = st.checkbox("Image no background", value= self.CTX.background,
                                     help = "Include the image no background file. Image")

        with col3:
            self.CTX.path_pos = st.checkbox("Path CooRecorder",value= self.CTX.path_pos,
                                        help="Include the path coorecorder files. Metrics>Path>Pos")
            self.CTX.ann_ew = st.checkbox("Annotation Earlywood",value=self.CTX.ann_ew,
                                 help = "Include the annotation earlywood file. Ring Editing>Annotation")

        with col4:
            self.CTX.area_csv = st.checkbox("Area csv", value = self.CTX.area_csv,
                                        help="Include the area csv files. Metrics>Area>CSV")
            self.CTX.ann_other = st.checkbox("Annotation Other",value=self.CTX.ann_other,
                                 help = "Include the annotation other file. Ring Editing>Annotation")


        if self.CTX.path_csv:
            files = Path(self.CTX.output_dir / "metrics/path").glob("*.csv")
            files_to_export += [f for f in files ]

        if self.CTX.area_metrics_report:
            file = Path(self.CTX.output_dir) / "metrics/metrics.pdf"
            files_to_export += [file]

        if self.CTX.image:
            file = Path(self.CTX.output_dir) / "image.png"
            files_to_export += [file]

        if self.CTX.path_images:
            files = Path(self.CTX.output_dir / "metrics/path").glob("*.png")
            files_to_export += [f for f in files ]

        if self.CTX.ann_lw:
            file = Path(self.CTX.output_dir) / "latewood_read.json"
            files_to_export += [file ]

        if self.CTX.background:
            file = Path(self.CTX.output_dir) / "background.png"
            files_to_export += [file ]

        if self.CTX.path_pos:
            files = Path(self.CTX.output_dir / "metrics/path").glob("*.pos")
            files_to_export += [f for f in files ]

        if self.CTX.ann_ew:
            file = Path(self.CTX.output_dir) / "earlywood_read.json"
            files_to_export += [file ]

        if self.CTX.area_csv:
            file = Path(self.CTX.output_dir) / "metrics/measurements.csv"
            files_to_export += [file]

        if self.CTX.ann_other:
            file = Path(self.CTX.output_dir) / "other_read.json"
            files_to_export += [file]




        return files_to_export

    def zip_files(self, files_to_export):
        tmp_dir = self.CTX.output_dir / "tmp"
        if tmp_dir.exists():
            os.system(f"rm -rf {tmp_dir}")
        tmp_dir.mkdir(exist_ok=True, parents=True)
        for file in files_to_export:
            os.system(f"cp {file} {tmp_dir}")

        zip_name = f"{self.CTX.code}.zip"
        zip_file_path = tmp_dir / zip_name

        # 1.1 zip the files
        if not zip_file_path.exists():
            os.system(f"cd {tmp_dir} && zip -r {zip_name} .")
        return zip_file_path, zip_name

    def download_results(self):
        #1.0 zip the results in self.CTX.output_dir
        files_to_export = self.generate_file_list_to_export()
        #1.0 copy files to tmp dir
        zip_file_path, zip_name = self.zip_files(files_to_export)

        #2.0 download the zip file
        args = dict(
            file_path = str(zip_file_path),
            label = "Download Results",
            filen_name = zip_name,
            mime = "application/zip"
        )
        download_button(**args)




def main(runtime_config_path):
    ui = UI(runtime_config_path)
    if check_image(ui.CTX):
        return

    ui.download_results()

    ui.CTX.save_config()

    return

