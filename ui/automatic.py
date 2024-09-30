import streamlit as st
import numpy as np
import cv2

from shapely.geometry import Polygon, Point
from streamlit_option_menu import option_menu
from pathlib import Path

from lib.image import LabelMeInterface as UserInterface, Drawing, Color
from lib.io import load_json, write_binary_file
from lib.inbd import INBD
from backend.labelme_layer import LabelmeShapeType, LoadLabelmeObject
from ui.common import Context, download_button

class LatewoodMethods:
    cstrd = "CS-TRD"
    inbd = "INBD"

class Shapes:
    pith = "Pith"
    latewood = "Latewood"
    earlywood = "Earlywood"
    knot = "Knot"

class Pith:
    pixel = "Pixel"
    boundary = "Boundary"
    manual = "Manual"
    automatic = "Automatic"
    apd = "APD"
    apd_pl = "APD-PL"
    apd_dl = "APD-DL"


class InbdModels:
    pinus = "Pinus Taeda"
    eh = "INBD-EH"

def annotate_pith():
    pass


class PithInterface(UserInterface):
    def __init__(self, image_path, output_json_path, output_image_path, pith_model=None):
        super().__init__(image_path, output_json_path)
        self.output_image_path = output_image_path
        self.pith_model = pith_model

    def parse_output(self):

        object = LoadLabelmeObject(self.output_path)
        if len(object.shapes) > 1:
            st.error("More than one shape found. Add only one shape")
            return None

        shape = object.shapes[0]
        if shape.shape_type != LabelmeShapeType.point and self.pith_model == Pith.pixel:
            st.error("Shape is not a point. Remember that you are using the pixel model")
            return None

        if shape.shape_type != LabelmeShapeType.polygon and self.pith_model == Pith.boundary:
            st.error("Shape is not a polygon")
            return None

        if shape.shape_type == LabelmeShapeType.point:
            return Point(shape.points[0])

        if shape.shape_type == LabelmeShapeType.polygon:
            return Polygon(shape.points)

        return None

    def generate_center_mask(self, output_path, results):
        if self.pith_model == Pith.pixel:
            mask = np.zeros(cv2.imread(self.image_path).shape[:2], dtype=np.uint8)
            x,y = results.xy
            x = int(x[0])
            y = int(y[0])
            mask[int(x), int(y)] = 255
            cv2.imwrite(output_path, mask)
            return

        image = cv2.imread(self.image_path)
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask = Drawing.fill(results.exterior, mask, Color.white, opacity=1)
        cv2.imwrite(output_path, mask)
        return



class ViewContext(Context):
    def init_specific_ui_components(self):
        config = self.config["image"]

        self.image_path = self.output_dir / config["image_path"]
        self.image_no_background_path = self.output_dir / config["background"]["image_path"]
        if Path(self.image_no_background_path).exists():
            self.image_orig_path = self.image_path
            self.image_path = self.image_no_background_path

        self.inbd_models = self.config["automatic"]["inbd_models"]
        self.model_path = self.config["automatic"]["model_path"]
        self.upload_model = self.config["automatic"]["upload_model"]
        self.pith_mask = self.config["automatic"]["pith_mask"]
        self.number_of_rays = self.config["automatic"]["number_of_rays"]

        return

    def update_config(self):
        self.config["automatic"]["model_path"] = str(self.model_path)
        self.config["automatic"]["upload_model"] = self.upload_model
        self.config["automatic"]["pith_mask"] = str(self.pith_mask)
        self.config["automatic"]["number_of_rays"] = self.number_of_rays

        return

    def reset_parameters(self):
        #TODO: Implement
        pass



class UI:

    def __init__(self, runtime_config_path):
        CTX = ViewContext(runtime_config_path)
        CTX.init_specific_ui_components()
        self.CTX = CTX

    def select_shape(self):
        selected = st.selectbox( "Select Shape", (Shapes.pith, Shapes.latewood, Shapes.earlywood, Shapes.knot))
        return selected

    def annotate_pith_manual(self, selected):
            annotate = st.button("Annotate")
            if annotate:
                self.CTX.pith_mask = self.CTX.output_dir / "pith_mask"
                self.CTX.pith_mask.mkdir(exist_ok=True, parents=True)
                self.CTX.pith_mask = self.CTX.output_dir / "pith_mask" / f"{self.CTX.image_orig_path.stem}.png"
                interface = PithInterface(self.CTX.image_orig_path, self.CTX.output_dir / "pith.json",
                                          self.CTX.output_dir / "pith.png",
                                          pith_model=selected)
                interface.interface()
                results = interface.parse_output()
                if results is None:
                    return
                interface.generate_center_mask(self.CTX.pith_mask, results)
                #display image mask
                st.image(cv2.imread(self.CTX.pith_mask), use_column_width=True)

    def shape_pith(self):
        pith_model = st.radio("Model", [Pith.pixel, Pith.boundary], horizontal=True)
        pith_method = st.radio("Method", [Pith.manual, Pith.automatic], horizontal=True)
        if pith_method == Pith.manual:
            self.annotate_pith_manual(pith_model)

        elif pith_model == Pith.pixel:
            #Pith model automatic
            pith_model = st.radio("Method", [Pith.apd, Pith.apd_pl, Pith.apd_dl], horizontal=True)
            run_button = st.button("Run")
            if run_button:
                # TODO: Implement
                st.write(f"Running {pith_model} METHOD")
                pass

    def cstrd_parameters(self):
        st.subheader("Parameters")
        sigma = st.slider("Sigma", 1.0, 10.0, 3.0)

        advanced = st.checkbox("Advanced parameters")
        if advanced:
            low = st.slider("Gradient threshold low", 0.0, 50.0, 5.0)
            high = st.slider("Gradient threshold high", 0.0, 50.0, 10.0)
            height = st.slider("Image Height", 0.0, 1500.0, 3000.0)
            width = st.slider("Image Width", 0.0, 1500.0, 3000.0)

    def cstrd_run(self):
        return "TODO.json"

    def inbd_parameters(self):
        self.output_dir_inbd = self.CTX.output_dir / "inbd"
        self.output_dir_inbd.mkdir(exist_ok=True, parents=True)
        output_model_path = f"{self.output_dir_inbd}/model.pt.zip"
        st.subheader("Parameters")
        self.CTX.upload_model = st.checkbox("Upload your own model", value=self.CTX.upload_model)
        model = st.radio("Model", [InbdModels.pinus, InbdModels.eh], horizontal=True, disabled=self.CTX.upload_model)
        if self.CTX.upload_model:
            output_model_path = file_model_uploader("Choose a file", output_model_path, ".pt.zip")
            st.warning("Remember that if you do not upload a model, a model uploaded in the pass will be used")
        else:
            output_model_path = self.CTX.inbd_models[model]

        # add input number option
        nr = st.number_input("Number of rays", 1, 1000, self.CTX.number_of_rays)
        if nr != self.CTX.number_of_rays:
            self.CTX.number_of_rays = nr
        self.CTX.model_path = output_model_path

        return

    def inbd_run(self):
        inbd = INBD(self.CTX.image_orig_path, self.CTX.pith_mask, Path(self.CTX.model_path), self.output_dir_inbd,
                    Nr=self.CTX.number_of_rays)
        results_path = inbd.run()
        return results_path

    def shape_latewood(self):
        method_latewood = st.radio("Method", [LatewoodMethods.cstrd, LatewoodMethods.inbd], horizontal=True)

        if method_latewood == LatewoodMethods.inbd:
            self.inbd_parameters()
        else:
            self.cstrd_parameters()

        st.divider()

        run_button = st.button("Run", use_container_width=True, disabled=not Path(self.CTX.model_path).exists()
                if method_latewood == LatewoodMethods.inbd else False )

        if run_button:
            results_path = self.inbd_run() if method_latewood == LatewoodMethods.inbd else self.cstrd_run()
            st.write("Results saved in: ", results_path)

            download_button(results_path, "Download", "results.json", "application/json")



def main(runtime_config_path):
    ui = UI(runtime_config_path)
    selected = ui.select_shape()
    st.divider()

    if selected == Shapes.pith:
        ui.shape_pith()

    if selected == Shapes.knot:
        #TODO: Implement
        run_button = st.button("Annotate")

    if selected == Shapes.latewood:
        ui.shape_latewood()

    st.divider()
    #save status
    ui.CTX.save_config()

    return

def file_model_uploader(label, output_file, extension):
    uploaded_cw_annotation_file = st.file_uploader(label, type=[extension])
    if uploaded_cw_annotation_file:
        write_binary_file(uploaded_cw_annotation_file.read(), output_file)


    return output_file




class PithBoundaryInterface(UserInterface):

    def __init__(self, image_path, output_file):
        super().__init__(image_path, output_file)

    def parse_output(self):
        try:
            data = load_json(self.output_path)
        except FileNotFoundError:
            st.write("No json file found")

        line = np.array(data['shapes'][0]['points'])
        pixels_length = int(np.linalg.norm(line[0] - line[1]))
        return pixels_length