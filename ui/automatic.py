import streamlit as st

from streamlit_option_menu import option_menu
from pathlib import Path

from lib.image import LabelMeInterface as UserInterface
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


class ViewContext(Context):
    def init_specific_ui_components(self):
        config = self.config["image"]

        self.image_path = self.output_dir / config["image_path"]
        self.image_no_background_path = self.output_dir / config["background"]["image_path"]
        if Path(self.image_no_background_path).exists():
            self.image_path = self.image_no_background_path


        return

    def update_config(self):


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
                    annotate_pith()

            if pith_method == Pith.automatic:
                selected = st.radio("Method", [Pith.apd, Pith.apd_pl, Pith.apd_dl], horizontal=True)
                run_button = st.button("Run")
                if run_button:
                    st.write(f"Running {selected} METHOD")
                    pass

        if selected == Pith.boundary:
            annotate = st.button("Annotate")
            if annotate:
                annotate_pith()


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
            st.divider()
            st.subheader("Parameters")
            model = st.radio("Model", [InbdModels.pinus, InbdModels.eh], horizontal=True)

            st.divider()

            run_button = st.button("Run", use_container_width=True)
            if run_button:
                # TODO: Implement
                st.write("Running INBD METHOD")
                save_button = st.button("Download")
                if save_button:
                    #TODO: Implement. https://docs.streamlit.io/develop/api-reference/widgets/st.download_button
                    pass









    st.divider()
    #save status
    CTX.save_config()

    return

def annotate_pith():
    #TODO: Implement
    pass




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