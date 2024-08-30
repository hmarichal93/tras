import os
from abc import ABC, abstractmethod


class UserInterface(ABC):
    def __init__(self, image_path, output_path):
        self.image_path = image_path
        self.output_path = output_path

    @abstractmethod
    def interface(self):
        pass





class LabelMeInterface(UserInterface):
    def __init__(self, image_path, output_path):
        super().__init__(image_path, output_path)

    def interface(self):
        command = f"labelme {self.image_path} -O {self.output_path}  --nodata "
        os.system(command)

    @abstractmethod
    def parse_output(self):
        pass
