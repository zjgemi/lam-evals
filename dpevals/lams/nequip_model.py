from pathlib import Path
from typing import List, Optional, Tuple

import ase
import numpy as np
from dpevals import LAM


class NequipModel(LAM):
    def __init__(self, model: Path, params: Optional[dict] = None):
        self.model = model
        self.params = params
        self.energy_bias = self.params.get("energy_bias", None)
        from nequip.ase import NequIPCalculator
        self.calc = NequIPCalculator.from_deployed_model(self.model, device="cuda")

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
