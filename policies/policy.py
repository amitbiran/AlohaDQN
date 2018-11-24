from abc import ABC, abstractmethod


class Policy(ABC):

    @abstractmethod
    def take_action(self, arr):
        pass