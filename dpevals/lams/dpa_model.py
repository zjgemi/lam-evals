import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from dpevals import LAM

type_map = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]


class DPAModel(LAM):
    def __init__(self, model: Path, params: Optional[dict] = None):
        self.model = model
        from deepmd_pt.infer.deep_eval import DeepPot
        self.dp = DeepPot(model)

    def evaluate(self,
                 coord: np.ndarray,
                 cell: Optional[np.ndarray],
                 atype: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        coord = coord.reshape([1, -1, 3])
        if cell is not None:
            cell = cell.reshape([1, 3, 3])
        atype = np.array([type_map.index(i) for i in atype])
        atype = atype.reshape([1, -1])
        e, f, v = self.dp.eval(coord, cell, atype, infer_batch_size=1)
        return e.reshape([1])[0], f.reshape([-1, 3]), v.reshape([3, 3])

    def validate(self,
                 systems: List[Path],
                 batch_size: Union[str, int] = "auto",
    ) -> Tuple[float, float, Optional[float]]:
        with open("valid.txt", "w") as f:
            f.write("\n".join([str(sys) for sys in systems]))
        args = ["dp_pt", "test", "-m", self.model, "-f", "valid.txt", "-n", "99999999", "-b", batch_size]
        print("Run command '%s'" % " ".join(args))
        with subprocess.Popen(args=args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            output = p.stdout.read().decode("utf-8")
            p.wait()
            ret_code = p.poll()
        print(output)
        assert ret_code == 0, "Command '%s' failed" % " ".join(args)
        rmse_e = rmse_f = rmse_v = None
        for line in output[output.rfind("----------weighted average of errors-----------"):].splitlines():
            if "rmse_e:" in line:
                rmse_e = float(line.split()[-1])
            if "rmse_f:" in line:
                rmse_f = float(line.split()[-1])
            if "rmse_v:" in line:
                rmse_v = float(line.split()[-1])
        return rmse_e, rmse_f, rmse_v
