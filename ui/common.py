from abc import ABC, abstractmethod
from pathlib import Path

from lib.io import load_json, write_json


class Shapes:
    pith = "Pith"
    latewood = "Late Wood"
    earlywood = "Early Wood"
    knot = "Knot"
    compresionwood = "Compression Wood"

class Color:
    red = "red"
    green = "green"
    blue = "blue"
    yellow = "yellow"
    black = "black"
    white = "white"

class Context(ABC):
    def __init__(self, runtime_config_path):
        #set general config
        config = load_json(runtime_config_path)
        self.output_dir = Path(config["general"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_config_path = runtime_config_path

        self.display_image_size = config["general"]["display_image_size"]

        self.config = config


    def save_config(self):
        self.update_config()
        write_json(self.config, self.runtime_config_path)

    @abstractmethod
    def init_specific_ui_components(self):
        pass


    @abstractmethod
    def update_config(self):
        pass
