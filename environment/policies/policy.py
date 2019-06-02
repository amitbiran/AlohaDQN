from abc import ABC, abstractmethod
"""
determin how a user may act.
in the base usecase of this project we didnt use this functionality 
may fits different usecases
"""

class Policy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def take_action(self, items):
        pass


    def on_finish_episode(self):
        pass