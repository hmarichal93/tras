import streamlit as st
import numpy as np
import cv2

from shapely.geometry import Polygon, Point
from streamlit_option_menu import option_menu
from pathlib import Path

from lib.image import LabelMeInterface as UserInterface, Drawing, Color
from lib.io import load_json, write_json, bytesio_to_dict, read_file_as_binary
from lib.inbd import INBD
from backend.labelme_layer import LabelmeShapeType, LoadLabelmeObject

from ui.common import Context

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


    return


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




def main(runtime_config_path):
    global CTX
    CTX = ViewContext(runtime_config_path)
    CTX.init_specific_ui_components()

    selected = st.selectbox( "Select Shape", (Shapes.pith, Shapes.latewood, Shapes.earlywood, Shapes.knot))
    reset_button = st.button("Reset Parameters")
    if reset_button:
        CTX.reset_parameters()
    st.divider()


    if selected == Shapes.pith:
        selected = st.radio("Model", [Pith.pixel, Pith.boundary], horizontal=True)
        if selected == Pith.pixel:
            pith_method = st.radio("Method", [Pith.manual, Pith.automatic], horizontal=True)
            if pith_method == Pith.manual:
                annotate = st.button("Annotate")
                if annotate:
                    interface = PithInterface(CTX.image_path, CTX.output_dir / "pith.json", CTX.output_dir / "pith.png",
                                              pith_model=Pith.pixel)
                    interface.interface()
                    results = interface.parse_output()
                    if results is None:
                        return
                    interface.generate_center_mask(CTX.output_dir / "pith_mask.png", results)
                    CTX.pith_mask = CTX.output_dir / "pith_mask.png"

            if pith_method == Pith.automatic:
                selected = st.radio("Method", [Pith.apd, Pith.apd_pl, Pith.apd_dl], horizontal=True)
                run_button = st.button("Run")
                if run_button:
                    st.write(f"Running {selected} METHOD")
                    pass

        if selected == Pith.boundary:
            pith_method = st.radio("Method", [Pith.manual, Pith.automatic], horizontal=True)
            if pith_method == Pith.manual:
                annotate = st.button("Annotate")
                if annotate:
                    CTX.pith_mask = CTX.output_dir / "pith_mask"
                    CTX.pith_mask.mkdir(exist_ok=True, parents=True)
                    CTX.pith_mask = CTX.output_dir / "pith_mask" / f"{CTX.image_orig_path.stem}.png"
                    interface = PithInterface(CTX.image_orig_path, CTX.output_dir / "pith.json", CTX.output_dir / "pith.png",
                                              pith_model=Pith.boundary)
                    interface.interface()
                    results = interface.parse_output()
                    if results is None:
                        return
                    interface.generate_center_mask(CTX.pith_mask, results)


    if selected == Shapes.knot:
        run_button = st.button("Annotate")


    if selected == Shapes.latewood:
        method_latewood = st.radio("Method", [LatewoodMethods.cstrd, LatewoodMethods.inbd], horizontal=True)
        if method_latewood == LatewoodMethods.cstrd:
            st.divider()

            st.subheader("Parameters")
            sigma = st.slider("Sigma", 1.0, 10.0, 3.0)

            advanced = st.checkbox("Advanced parameters")
            if advanced:
                low = st.slider("Gradient threshold low", 0.0, 50.0, 5.0)
                high = st.slider("Gradient threshold high", 0.0, 50.0, 10.0)
                height = st.slider("Image Height", 0.0, 1500.0, 3000.0)
                width = st.slider("Image Width", 0.0, 1500.0, 3000.0)

            st.divider()

            run_button = st.button("Run", use_container_width=True)
            if run_button:
                #TODO: Implement
                st.write("Running CS-TRD METHOD")
                save_button = st.button("Download")
                if save_button:
                    # TODO: Implement. https://docs.streamlit.io/develop/api-reference/widgets/st.download_button
                    pass

        if method_latewood == LatewoodMethods.inbd:
            output_dir_inbd = CTX.output_dir / "inbd"
            output_dir_inbd.mkdir(exist_ok=True, parents=True)
            output_model_path = f"{output_dir_inbd}/model.pt.zip"

            st.divider()
            st.subheader("Parameters")
            CTX.upload_model = st.checkbox("Upload your own model", value = CTX.upload_model)
            model = st.radio("Model", [InbdModels.pinus, InbdModels.eh], horizontal=True, disabled =  CTX.upload_model)
            if CTX.upload_model:
                output_model_path = file_model_uploader("Choose a file", output_model_path, ".pt.zip")
                st.warning("Remember that if you do not upload a model, a model uploaded in the pass will be used")
            else:
                output_model_path = CTX.inbd_models[model]

            #add input number option
            nr = st.number_input("Number of rays", 1, 1000, CTX.number_of_rays)
            if nr != CTX.number_of_rays:
                CTX.number_of_rays = nr
            CTX.model_path = output_model_path
            st.divider()

            run_button = st.button("Run", use_container_width=True, disabled=not Path(CTX.model_path).exists())
            if run_button:
                inbd = INBD(CTX.image_orig_path, CTX.pith_mask, Path(CTX.model_path), output_dir_inbd, Nr=nr)
                results_path = inbd.run()
                st.write("Results saved in: ", results_path)
                #download results_path (json file). Using thte download button
                #read json file and convert to binary
                json_content = read_file_as_binary(results_path)
                st.download_button(label="Download", data=json_content, file_name="results.json", mime="application/json")



    st.divider()
    #save status
    CTX.save_config()

    return

def file_model_uploader(label, output_file, extension):
    uploaded_cw_annotation_file = st.file_uploader(label, type=[extension])
    if uploaded_cw_annotation_file:
        with open(output_file, "wb") as f:
            f.write(uploaded_cw_annotation_file.read())

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