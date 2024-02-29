import abc
from abc import ABC
from typing import Dict

from dpevals import LAM


class Eval(ABC):
    def __init__(self,
                 lam: LAM,
    ):
        self.lam = lam

    @abc.abstractmethod
    def run(self) -> Dict[str, float]:
        raise NotImplementedError()
