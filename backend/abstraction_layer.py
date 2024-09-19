from abc import ABC, abstractmethod

class UserInterface(ABC):
    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def write(self, data):
        pass


