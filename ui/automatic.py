import streamlit as st
import numpy as np
import cv2
import os
import torch

from shapely.geometry import Polygon, Point
from pathlib import Path

from lib.image import  Drawing, Color, load_image, write_image
from lib.io import load_json, write_binary_file
from lib.inbd import INBD
from lib.cstrd import CSTRD
from lib.deepcstrd import DEEPCSTRD, get_model_path, DeepCSTRD_MODELS
from lib.apd import APD

from backend.labelme_layer import (LabelmeShapeType, LabelmeObject, AL_LateWood_EarlyWood,
                                   LabelmeInterface as UserInterface, ring_relabelling)
from ui.common import Context, download_button, RunningWidget, Pith, display_image, check_image, resize_slider

class LatewoodMethods:
    cstrd = "CS-TRD"
    inbd = "INBD"
    deep_cstrd = "Deep CS-TRD"

class Shapes:
    pith = "Pith"
    latewood = "Latewood"
    earlywood = "Earlywood"
    knot = "Knot"


class InbdModels:
    pinus = "Pinus Taeda"
    eh = "INBD-EH"

def annotate_pith():
    pass


class PithInterface(UserInterface):
    def __init__(self, image_path, output_json_path, output_image_path, pith_model=None):
        super().__init__(read_file_path=image_path, write_file_path=output_json_path)
        self.output_image_path = output_image_path
        self.pith_model = pith_model

    def parse_output(self):

        object = LabelmeObject(self.write_file_path)
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

    def automatic(self, pith_model,
                  weights_path="automatic_methods/pith_detection/apd/checkpoints/yolo/all_best_yolov8.pt",
                  output_dir=None,
                  params_dict= None):

        if params_dict is None:
            params_dict = {}

        lo_w = params_dict.get("lo_w", 3)
        st_w = params_dict.get("st_w", 3)
        st_sigma = params_dict.get("st_sigma", 1.2)
        percent_lo = params_dict.get("percent_lo", 0.7)
        resize = params_dict.get("resize", 1)

        apd = APD(self.read_file_path, self.write_file_path, method=pith_model, weights_path=weights_path,
                  output_dir= output_dir, percent_lo=percent_lo, st_w=st_w,
                  lo_w=lo_w, st_sigma=st_sigma, resize_factor=resize)
        status = apd.run()

        return status

    def generate_center_mask(self, output_path, results):
        if self.pith_model == Pith.pixel:
            mask = np.zeros(load_image(self.read_file_path).shape[:2], dtype=np.uint8)
            x,y = results.xy
            x = int(x[0])
            y = int(y[0])
            mask[int(x), int(y)] = 255
            write_image(output_path, mask)
            return

        image = load_image(self.read_file_path)
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask = Drawing.fill(results.exterior, mask, Color.white, opacity=1)
        write_image(str(output_path), mask)
        return

    def from_structure_to_labelme_shape(self, structure_list):
        pass

    def from_labelme_shape_to_structure(self, shapes):
        pass

class ViewContext(Context):
    def init_specific_ui_components(self):
        config = self.config["image"]

        self.image_path = self.output_dir / config["image_path"]
        self.image_no_background_path = self.output_dir / config["background"]["image_path"]
        self.json_background_path = self.output_dir / config["background"]["json_path"]
        if Path(self.image_no_background_path).exists():
            self.image_orig_path = self.image_path
            self.image_path = self.image_no_background_path
        else:
            self.image_orig_path = self.image_path
            self.image_no_background_path = self.image_path

        self.autocomplete_ring_date = config["metadata"]["autocomplete_ring_date"]
        self.harvest_date = int(config["metadata"]["harvest_date"]["year"])

        self.inbd_models = self.config["automatic"]["inbd_models"]
        self.model_path = self.config["automatic"]["model_path"]
        self.upload_model = self.config["automatic"]["upload_model"]
        self.pith_mask_path = self.config["automatic"]["pith_mask"]
        self.number_of_rays = self.config["automatic"]["number_of_rays"]
        self.inbd_resize_factor = self.config["automatic"]["inbd_resize_factor"]
        self.pith_model = self.config["automatic"]["pith_model"]
        self.pith_mode = self.config["automatic"]["pith_mode"]
        self.pith_method = self.config["automatic"]["pith_method"]

        self.sigma = self.config["automatic"]["sigma"]
        self.th_low = self.config["automatic"]["th_low"]
        self.th_hight = self.config["automatic"]["th_hight"]
        lw_path = self.config["manual"]["annotations_files"]["late_wood"]
        self.lw_annotations = self.output_dir / "none.json" if isinstance(lw_path, list) else Path(lw_path)

        self.apd_params = self.config["automatic"]["apd_params"]
        self.code = self.config["image"]["metadata"]["code"]

        self.deep_cstrd = self.config["automatic"]["deep_cstrd"]

        return

    def update_config(self):
        self.config["automatic"]["pith_model"] = self.pith_model
        self.config["automatic"]["pith_mode"] = self.pith_mode
        self.config["automatic"]["model_path"] = str(self.model_path)
        self.config["automatic"]["upload_model"] = self.upload_model
        self.config["automatic"]["pith_mask"] = str(self.pith_mask_path)
        self.config["automatic"]["number_of_rays"] = self.number_of_rays
        self.config["automatic"]["inbd_resize_factor"] = self.inbd_resize_factor
        self.config["automatic"]["sigma"] = self.sigma
        self.config["automatic"]["th_low"] = self.th_low
        self.config["automatic"]["th_hight"] = self.th_hight
        self.config["automatic"]["apd_params"] = self.apd_params
        self.config["automatic"]["pith_method"] = self.pith_method
        self.config["automatic"]["deep_cstrd"] = self.deep_cstrd


        return

    def reset_parameters(self):
        #TODO: Implement
        pass



class UI:

    def __init__(self, runtime_config_path):
        st.header("Ring Detection")
        st.markdown(
            """
            Three methods are available for the automatic detection of rings: CS-TRD, DeepCS-TRD and INBD. Pith annotations is required.
            """
        )
        st.divider()

        CTX = ViewContext(runtime_config_path)
        CTX.init_specific_ui_components()
        self.CTX = CTX

    def select_shape(self):
        selected = st.selectbox( "Select Shape", (Shapes.pith, Shapes.latewood, Shapes.earlywood, Shapes.knot))
        return selected

    def get_pith(self, pith_model, pith_method):
        if pith_method == Pith.automatic:
            advanced = st.checkbox("Advanced parameters", value=False)
            if advanced:
                percent_lo = st.slider("Percent Lo", 0.0, 1.0, float(self.CTX.apd_params["percent_lo"]), step=0.1)
                st_w = st.slider("ST Window", 1, 10,  int(self.CTX.apd_params["st_w"]))
                lo_w = st.slider("LO Window", 1, 10,  int(self.CTX.apd_params["lo_w"]))
                st_sigma = st.slider("ST Sigma", 0.1, 10.0,  float(self.CTX.apd_params["st_sigma"]), step=0.1)
                resize = resize_slider(default=  int(self.CTX.apd_params["resize"]), legend="Resize", help="Resize factor for the image.\n",
                                       max=10, min=1)


                params_dict = {"percent_lo": percent_lo, "st_w": st_w, "lo_w": lo_w,
                               "st_sigma": st_sigma, "resize": resize}

                self.CTX.apd_params = params_dict
            else:
                params_dict = self.CTX.apd_params

        annotate = st.button("Run")
        if annotate:
            gif_runner = RunningWidget()

            self.CTX.pith_mask = self.CTX.output_dir / "pith_mask"
            self.CTX.pith_mask.mkdir(exist_ok=True, parents=True)
            self.CTX.pith_mask_path = self.CTX.output_dir / "pith_mask" / f"{self.CTX.image_orig_path.stem}.png"
            interface = PithInterface(self.CTX.image_orig_path, self.CTX.output_dir / "pith.json",
                                      self.CTX.output_dir / "pith.png",
                                      pith_model=pith_model)

            if pith_method == Pith.manual:
                interface.interface()
                status = True
            else:
                status = interface.automatic(pith_model, output_dir = self.CTX.pith_mask, params_dict = params_dict)

            results = interface.parse_output() if status else None

            if results is None:
                st.write("No results found")
                return
            interface.generate_center_mask(self.CTX.pith_mask_path, results)
            #display image mask
            mask  = load_image(self.CTX.pith_mask_path)
            #convert to gray scale
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            image = load_image(self.CTX.image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image[mask > 0] = Color.red

            gif_runner.empty()
            display_image(image)

        return

    def shape_pith(self):
        model = [Pith.pixel, Pith.boundary]
        pith_model = st.radio("Model", model , horizontal=True, index = model.index(self.CTX.pith_model))
        if pith_model != self.CTX.pith_model:
            self.CTX.pith_model = pith_model
        mode = [Pith.manual, Pith.automatic]
        pith_mode = st.radio("Method", mode, horizontal=True, index = mode.index(self.CTX.pith_mode))
        if pith_mode != self.CTX.pith_mode:
            self.CTX.pith_mode = pith_mode

        if pith_mode == Pith.manual:
            self.get_pith(pith_model, pith_mode)
        else:
            #Pith model automatic
            pith_methods = [Pith.apd, Pith.apd_pl, Pith.apd_dl]
            pith_method = st.radio("Method", pith_methods, horizontal=True, index = pith_methods.index(self.CTX.pith_method))
            if pith_method != self.CTX.pith_method:
                self.CTX.pith_method = pith_method
            self.get_pith(pith_method, pith_mode)

    def cstrd_parameters(self):
        st.subheader("Parameters")
        sigma = st.slider("Sigma", 1.0, 10.0, float(self.CTX.sigma), step=0.1, help="Sigma for the gaussian filter")
        if sigma != self.CTX.sigma:
            self.CTX.sigma = sigma
        advanced = st.checkbox("Advanced parameters")
        if advanced:
            low = st.slider("Gradient threshold low", 0.0, 50.0, float(self.CTX.th_low))
            if low != self.CTX.th_low:
                self.CTX.th_low = low
            high = st.slider("Gradient threshold high", 0.0, 50.0, float(self.CTX.th_hight))
            if high != self.CTX.th_hight:
                self.CTX.th_hight = high


    def deep_cstrd_parameters(self):
        self.output_dir_deepcstrd = self.CTX.output_dir / "deepcstrd"
        self.output_dir_deepcstrd.mkdir(exist_ok=True, parents=True)
        output_model_path = f"{self.output_dir_deepcstrd}/model.pth"

        st.subheader("Parameters")
        config = self.CTX.deep_cstrd
        config['upload_model'] = st.checkbox("Upload your own model", value=config['upload_model'])
        models_list = [DeepCSTRD_MODELS.pinus_v1, DeepCSTRD_MODELS.pinus_v2, DeepCSTRD_MODELS.gleditsia,
                       DeepCSTRD_MODELS.salix, DeepCSTRD_MODELS.all]
        model = st.radio("Model", models_list, horizontal=True,
                         index=config.get("model_id"))
        if config['upload_model']:
            output_model_path = file_model_uploader("Choose a file", output_model_path, ".pth")
            st.warning("Remember that if you do not upload a model, a model uploaded in the pass will be used")
        else:
            if model != config.get("model_id"):
                self.CTX.deep_cstrd["model_id"] = models_list.index(model)
                self.deep_cstrd_model = model

        advanced = st.checkbox("Advanced parameters")
        if advanced:
            alpha = st.slider("Angle α (degrees)", 0, 89, config.get("alpha"), 5)
            tile_sizes = [0, 64, 128, 256, 512]
            tile_size = st.selectbox("Tile Size", tile_sizes, index=tile_sizes.index(config.get("tile_size")))
            prediction_map_threshold = st.slider("Prediction Map Threshold", 0.0, 1.0, config.get("prediction_map_th"), 0.1)
            total_rotations = st.slider("Total Rotations", 0, 8, config.get("total_rotations"), 1)

            if alpha != config.get("alpha"):
                self.CTX.deep_cstrd["alpha"] = alpha
            if tile_size != config.get("tile_size"):
                self.CTX.deep_cstrd["tile_size"] = tile_size
            if prediction_map_threshold != config.get("prediction_map_th"):
                self.CTX.deep_cstrd["prediction_map_th"] = prediction_map_threshold
            if total_rotations != config.get("total_rotations"):
                self.CTX.deep_cstrd["total_rotations"] = total_rotations




    def deep_cstrd_run(self):
        self.output_dir_deepcstrd = self.CTX.output_dir / "deepcstrd"
        gif_runner = RunningWidget()
        try:
            self.CTX.deepcstrd_model_path = f"{self.output_dir_deepcstrd }/model.pth" \
                    if self.CTX.deep_cstrd['upload_model'] else (
                    get_model_path(self.deep_cstrd_model, self.CTX.deep_cstrd["tile_size"]))

            method = DEEPCSTRD(self.CTX.image_no_background_path, self.CTX.pith_mask_path,
                           Path(self.CTX.deepcstrd_model_path),
                           self.output_dir_deepcstrd,
                           Nr=self.CTX.number_of_rays, resize_factor=self.CTX.inbd_resize_factor,
                           background_path=self.CTX.json_background_path,
                           sigma=self.CTX.sigma,
                            alpha=self.CTX.deep_cstrd["alpha"],
                               weights_path=self.CTX.deepcstrd_model_path,
                               tile_size=self.CTX.deep_cstrd["tile_size"],
                               prediction_map_threshold=self.CTX.deep_cstrd["prediction_map_th"],
                               total_rotations=self.CTX.deep_cstrd["total_rotations"])

            results_path = method.run()
        except torch.OutOfMemoryError:
            st.error("Out of memory. Issue may happen because there is not enough GPU memory. "
                     "Try to increase the resize factor")
            results_path = None

        except AttributeError:
            st.error("Please remove background from image in the Image menu")
            results_path = None


        gif_runner.empty()
        return results_path
    def cstrd_run(self, lw_annotations=None):
        self.output_dir_cstrd = self.CTX.output_dir / "cstrd"
        os.system(f"rm -rf {self.output_dir_cstrd}")
        self.output_dir_cstrd.mkdir(exist_ok=True, parents=True)

        gif_runner = RunningWidget()

        method = CSTRD(self.CTX.image_no_background_path, self.CTX.pith_mask_path, Path(self.CTX.model_path), self.output_dir_cstrd,
                    Nr=self.CTX.number_of_rays, resize_factor=self.CTX.inbd_resize_factor,
                    background_path=self.CTX.json_background_path, sigma=self.CTX.sigma, th_low=self.CTX.th_low,
                    th_hight=self.CTX.th_hight,
                    gt_ring_json=lw_annotations,
                    include_gt_rings_in_output= True if lw_annotations is not None else False)

        results_path = method.run()
        gif_runner.empty()
        return results_path

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

        self.CTX.model_path = output_model_path


        return

    def inbd_run(self):
        gif_runner = RunningWidget()
        inbd = INBD(self.CTX.image_no_background_path, self.CTX.pith_mask_path, Path(self.CTX.model_path), self.output_dir_inbd,
                    Nr=self.CTX.number_of_rays, resize_factor=self.CTX.inbd_resize_factor,
                    background_path=self.CTX.json_background_path)
        results_path = inbd.run()
        gif_runner.empty()
        return results_path

    def parameters_latewood(self, method_latewood):
        if method_latewood == LatewoodMethods.inbd:
            self.inbd_parameters()
        elif method_latewood == LatewoodMethods.cstrd:
            self.cstrd_parameters()
        elif method_latewood == LatewoodMethods.deep_cstrd:
            self.deep_cstrd_parameters()
        else:
            st.error("Method not implemented")
            return
        # add input number option
        nr = st.number_input("Number of boundary points", 1, 1000, self.CTX.number_of_rays,
                             help="Number of points in each output boundary ring")
        if nr != self.CTX.number_of_rays:
            self.CTX.number_of_rays = nr

        resize_factor = resize_slider(default=int(self.CTX.inbd_resize_factor), legend="Resize Factor",
                               help="Resize factor for the image.\n""Be aware that the image will \n"
                               "be resized, which means that the \n""automatic method will work at a \n"
                               "lower resolution", max=10, min=1)

        if resize_factor != self.CTX.inbd_resize_factor:
            self.CTX.inbd_resize_factor = resize_factor
        return

    def shape_earlywood(self):
        if not (self.CTX.output_dir / "pith.json").exists():
            st.error("Pith annotation is required. Go to pith detection menu")
            return

        method_latewood = st.radio("Method", [LatewoodMethods.cstrd], horizontal=True)
        self.parameters_latewood(method_latewood)
        st.divider()
        #upload file
        disabled = not self.CTX.lw_annotations.exists()

        # if st.checkbox("Upload latewood annotations"):
        #     file_uploader("Upload latewood annotations", self.CTX.lw_annotations, ".json")
        # else:
        st.write("Latewood annotations file: ", self.CTX.lw_annotations)

        run_button = st.button("Run", use_container_width=True, disabled= disabled)

        if run_button:
            results_path = self.cstrd_run(str(self.CTX.lw_annotations))
            st.write("Results saved in: ", results_path)
            download_button(results_path, "Download", f"{self.CTX.code}_ew.json",
                            "application/json")
            image_draw_path = results_path.parent / "contours.png"
            #st.image(cv2.cvtColor(load_image(image_draw_path),cv2.COLOR_BGR2RGB), use_column_width=True)
            display_image((cv2.cvtColor(load_image(image_draw_path),cv2.COLOR_BGR2RGB)))



        return

    def display_results(self, results_path):
        image_contour_path = Path(results_path).parent / "contours.png" #if method_latewood == LatewoodMethods.inbd else None

        #display image in image_contour_path
        if image_contour_path is not None:
            rings_list = AL_LateWood_EarlyWood(results_path, None).read()
            image = load_image(self.CTX.image_path)
            #convert image to rgb
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for ring in rings_list:
                poly = Polygon(ring.points.tolist())
                image = Drawing.curve( poly.exterior.coords, image, Color.red, thickness=5)

            #st.image(image, use_column_width=True)
            display_image(image)
        return

    def shape_latewood(self):
        if not (self.CTX.output_dir / "pith.json").exists():
            st.error("Pith annotation is required. Go to pith detection menu")
            return

        method_latewood = st.radio("Method", [LatewoodMethods.cstrd, LatewoodMethods.deep_cstrd, LatewoodMethods.inbd],
                                   horizontal=True)
        self.parameters_latewood(method_latewood)
        st.divider()

        run_button = st.button("Run", use_container_width=True, disabled=not Path(self.CTX.model_path).exists()
                if method_latewood == LatewoodMethods.inbd else False )

        if run_button:
            if method_latewood == LatewoodMethods.inbd:
                results_path = self.inbd_run()
            elif method_latewood == LatewoodMethods.cstrd:
                results_path = self.cstrd_run()
            elif method_latewood == LatewoodMethods.deep_cstrd:
                results_path = self.deep_cstrd_run()
            else:
                st.error("Method not implemented")
                return


            if results_path is None:
                return

            #relabel rings
            if self.CTX.autocomplete_ring_date:
                ring_relabelling(self.CTX.image_path, results_path, self.CTX.harvest_date)
            st.write("Results saved in: ", results_path)
            output_name = f"{self.CTX.code}_lw_{method_latewood}.json"
            download_button(results_path, "Download", output_name,
                            "application/json")
            self.display_results(results_path)




def main(runtime_config_path):
    ui = UI(runtime_config_path)
    if check_image(ui.CTX):
        return
    selected = ui.select_shape()
    st.divider()

    if selected == Shapes.pith:
        ui.shape_pith()

    if selected == Shapes.knot:
        #TODO: Implement
        run_button = st.button("Annotate")

    if selected == Shapes.latewood:
        ui.shape_latewood()

    if selected == Shapes.earlywood:
        ui.shape_earlywood()

    st.divider()
    #export status
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