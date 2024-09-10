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
    def __init__(self, image_path, output_path, edit=False):
        super().__init__(image_path, output_path)
        self.edit = edit

    def interface(self):
        if self.edit:
            command = f"labelme {self.output_path}"

        else:
            command = f"labelme {self.image_path} -O {self.output_path}  --nodata "

        print(command)
        os.system(command)

    @abstractmethod
    def parse_output(self):
        pass

