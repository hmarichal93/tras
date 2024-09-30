import os

from ui.common import Context, download_button





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
        pass

    def update_config(self):
        pass





class UI:

    def __init__(self, runtime_config_path):
        CTX = ViewContext(runtime_config_path)
        CTX.init_specific_ui_components()
        self.CTX = CTX

    def download_results(self):
        #1.0 zip the results in self.CTX.output_dir
        zip_file_path = self.CTX.output_dir / "results.zip"
        if not zip_file_path.exists():
            os.system(f"cd {self.CTX.output_dir} && zip -r results.zip .")
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

    ui.download_results()

    ui.CTX.save_config()

    return

