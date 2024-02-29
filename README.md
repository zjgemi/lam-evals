# OpenLAM Evals

OpenLAM Evals is a framework for evaluating LAMs and LAM systems, and an open-source registry of benchmarks.

## LAM

A LAM should implement the `LAM` interface:

```python
class MyLAM(LAM):
    def __init__(self, model: Path, params: Optional[dict] = None):
        pass

    def evaluate(self,
                 coord: np.ndarray,
                 cell: Optional[np.ndarray],
                 atype: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        pass
```

## Eval

A eval case generally involves the following process: loading the dataset, using the LAM to perform inference on the dataset, evaluating the LAM's performance and returning a score.

```python
class MyEval(Eval):
    def __init__(self, lam: LAM, **kwargs):
        self.lam = lam
        pass

    def run(self) -> Dict[str, float]:
        pass
```