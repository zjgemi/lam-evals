import abc
from abc import ABC
from pathlib import Path
from typing import List, Optional, Tuple, Union

import dpdata
import numpy as np


class LAM(ABC):
    @abc.abstractclassmethod
    def __init__(self, model: Path, params: Optional[dict] = None):
        pass

    @abc.abstractclassmethod
    def evaluate(self,
                 coord: np.ndarray,
                 cell: Optional[np.ndarray],
                 atype: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        pass

    def validate(self,
                 systems: List[Path],
                 batch_size: Union[str, int] = "auto",
    ) -> Tuple[float, float, Optional[float]]:
        sum_err_e = sum_err_f = sum_err_v = 0.
        virial = False
        sum_natoms = 0
        sum_nframes = 0
        for sys in systems:
            mixed_type = len(list(sys.glob("*/real_atom_types.npy"))) > 0
            d = dpdata.MultiSystems()
            if mixed_type:
                d.load_systems_from_file(sys, fmt="deepmd/npy/mixed")
            else:
                k = dpdata.LabeledSystem(sys, fmt="deepmd/npy")
                d.append(k)
            for k in d:
                for i in range(len(k)):
                    cell = k[i].data["cells"][0]
                    if k[i].nopbc:
                        cell = None
                    coord = k[i].data["coords"][0]
                    force0 = k[i].data["forces"][0]
                    energy0 = k[i].data["energies"][0]
                    virial0 = k[i].data["virials"][0] if "virials" in k[i].data else None
                    ori_atype = k[i].data["atom_types"]
                    anames = k[i].data["atom_names"]
                    atype = np.array([anames[j] for j in ori_atype])
                    e, f, v = self.evaluate(coord, cell, atype)
                    n = force0.shape[0]
                    sum_err_e += np.sum(((energy0-e)/n)**2)
                    sum_err_f += np.sum((force0-f)**2)
                    if virial0 is not None and v is not None:
                        virial = True
                        sum_err_v += np.sum(((virial0-v)/n)**2)
                    sum_natoms += n
                    sum_nframes += 1
        rmse_e = np.sqrt(sum_err_e/sum_nframes)
        rmse_f = np.sqrt(sum_err_f/sum_natoms/3)
        rmse_v = np.sqrt(sum_err_v/sum_nframes/9) if virial else None
        return rmse_e, rmse_f, rmse_v
