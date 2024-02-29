import json
from pathlib import Path
from typing import List, Optional, Tuple

import ase
import numpy as np
import yaml
from dpevals import LAM


class OCPModel(LAM):
    def __init__(self, model: Path, params: Optional[dict] = None):
        self.model = model
        self.params = params

        params = self.params.copy()
        if "scale_dict" in params:
            with open("scale.json", "w") as f:
                json.dump(params.pop("scale_dict"), f)
        self.energy_bias = params.pop("energy_bias", None)
        with open("input.yaml", "w") as f:
            f.write(yaml.dump(params))
        from ocpmodels.common.relaxation.ase_utils import OCPCalculator
        self.calc = OCPCalculator("input.yaml", self.model, "forces", cpu=False)

    def evaluate(self,
                 coord: np.ndarray,
                 cell: Optional[np.ndarray],
                 atype: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        atoms = ase.Atoms(symbols=atype, positions=coord, cell=cell, pbc=True)
        atoms.calc = self.calc
        e = atoms.get_potential_energy()
        if self.energy_bias is not None:
            e += sum([self.energy_bias.get(a, 0.0) for a in atype])
        f = atoms.get_forces()
        return e, f, None
