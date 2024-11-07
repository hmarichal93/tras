import os
import streamlit as st

from streamlit_option_menu import option_menu
from PIL import Image
from pathlib import Path

from ui.image import main as image
from ui.automatic import main as automatic_ring_delineation
from ui.home import main as home
from ui.manual import main as manual
from ui.metrics import main as metrics
from ui.report import main as export
from ui.update import pull_last_changes_from_remote_repo


APP_NAME = "TRAS: An Interactive Sofware for tracing Tree Ring Cross Sections"
DEFAULT_CONFIG_PATH = "./config/default.json"
RUNTIME_CONFIG_PATH = "./config/runtime.json"


class Menu:
    home = "Home"
    image = "Image"
    automatic_ring_delineation = "Ring Detection"
    manual_ring_delineation = "Ring Editing"
    metrics = "Metrics"
    export = "Export"


def initialization():
    if Path(RUNTIME_CONFIG_PATH).exists():
        os.system(f"rm -rf {RUNTIME_CONFIG_PATH}")

    return


class Mode:
    single = "Single"
    batch = "Batch"



im = Image.open('assets/pixels_wood.jpg')
st.set_page_config( page_title=APP_NAME, page_icon=im, layout='wide')

def app_title_decorator(app_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Establecer el título de la aplicación con HTML personalizado
            st.markdown(f"""
                <h1 style="color: #A0522D; font-size: 40px; text-align: center;">
                    {app_name}
                </h1>
                <hr style="border: 2px solid #A0522D;">
            """, unsafe_allow_html=True)

            # Estilos CSS para ajustar el ancho de la barra lateral y el contenido principal
            st.markdown(
                """
                <style>
                /* Cambia el ancho de la barra lateral */
                [data-testid="stSidebar"] {
                    width: 33.33% !important;
                }
                /* Ajusta el contenido principal para que se adapte al ancho restante */
                [data-testid="stSidebar"] + div {
                    width: 66.67%;
                    margin-left: 33.33%;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            # Ejecutar la función original
            return func(*args, **kwargs)
        return wrapper
    return decorator
@app_title_decorator(APP_NAME)
def main():
    # 1. as sidebar menu
    with st.sidebar:
        selected = option_menu("", [Menu.home, Menu.image, Menu.automatic_ring_delineation,
                                    Menu.manual_ring_delineation, Menu.metrics, Menu.export], menu_icon="cast",
                               default_index=0)

    if selected == Menu.home:
        home(DEFAULT_CONFIG_PATH, RUNTIME_CONFIG_PATH)

    elif selected == Menu.image:
        image(RUNTIME_CONFIG_PATH)

    elif selected == Menu.automatic_ring_delineation:
        automatic_ring_delineation(RUNTIME_CONFIG_PATH)

    elif selected == Menu.manual_ring_delineation:
        manual(RUNTIME_CONFIG_PATH)

    elif selected == Menu.metrics:
        metrics(RUNTIME_CONFIG_PATH)

    elif selected == Menu.export:
        export(RUNTIME_CONFIG_PATH)

    with st.sidebar:

        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <!-- Enlace de GitHub -->
                    <a href="https://github.com/hmarichal93/dendrotool" target="_blank">
                        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="30" height="30">
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Botón de Streamlit (el verdadero funcional)
        if st.button("Update", key="streamlit_button", help="Update the app to the latest version"):
            pull_last_changes_from_remote_repo(os.path.dirname(os.path.abspath(__file__)))

        st.image("assets/wood_image.jpeg")


if __name__ == "__main__":
    main()
    #TODO: Add CooRecorder export to the app.
    # https://github.com/Gregor-Mendel-Institute/TRG-ImageProcessing/blob/826736d3012a2322130850e34c7bfdc873dfef42/CoreProcessingPipelineScripts/CNN/Mask_RCNN/postprocessing/postprocessingCracksRings.py#L779
