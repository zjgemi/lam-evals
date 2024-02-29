from pathlib import Path
from typing import Dict, Union

from dpevals import Eval, LAM


class EFVRMSE(Eval):
    def __init__(self, lam: LAM, dataset: Path, batch_size: Union[str, int] = "auto"):
        super().__init__(lam)
        self.dataset = dataset
        self.batch_size = batch_size

    def run(self) -> Dict[str, float]:
        systems = [f.parent for f in self.dataset.rglob("type.raw")]
        rmse_e, rmse_f, rmse_v = self.lam(systems, self.batch_size)
        return {
            "rmse_e": rmse_e,
            "rmse_f": rmse_f,
            "rmse_v": rmse_v,
        }
