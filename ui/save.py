import os

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

    def update_config(self):
        pass





class UI:

    def __init__(self, runtime_config_path):
        CTX = ViewContext(runtime_config_path)
        CTX.init_specific_ui_components()
        self.CTX = CTX

    def download_results(self):
        #1.0 zip the results in self.CTX.output_dir
        files_to_export = (list(self.CTX.output_dir.glob("*.json")) +
                           [self.CTX.output_dir / "metrics" / "measurements.csv"] +
                           [self.CTX.output_dir / "metrics" / "coorecorder.csv"] +
                           [self.CTX.output_dir / "metrics" / "rings.png"] +
                           [self.CTX.output_dir / "metrics" / "metrics.pdf"] +
                           [self.CTX.output_dir / "image.png"])

        #1.0 copy files to tmp dir
        tmp_dir = self.CTX.output_dir / "tmp"
        if tmp_dir.exists():
            os.system(f"rm -rf {tmp_dir}")
        tmp_dir.mkdir(exist_ok=True, parents=True)
        for file in files_to_export:
            os.system(f"cp {file} {tmp_dir}")
        zip_file_path = tmp_dir / "results.zip"

        #1.1 zip the files
        if not zip_file_path.exists():
            os.system(f"cd {tmp_dir} && zip -r results.zip .")
        #2.0 download the zip file
        args = dict(
            file_path = str(zip_file_path),
            label = "Download Results",
            filen_name = "results.zip",
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

