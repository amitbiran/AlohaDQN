from abc import ABC, abstractmethod


class Policy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def take_action(self, items):
        pass


    def on_finish_episode(self):
        pass