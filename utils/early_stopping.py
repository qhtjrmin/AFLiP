import torch.nn as nn

class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0, mode: str = "max"):
        assert mode in ("max", "min")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_metric = None
        self.best_epoch = -1
        self.best_model_state = None
        self.best_pred_state = None

        self.num_bad_epochs = 0

    def _is_better(self, metric: float) -> bool:
        if self.best_metric is None:
            return True

        if self.mode == "max":
            return metric > self.best_metric + self.min_delta
        else:  # "min"
            return metric < self.best_metric - self.min_delta

    def step(
        self,
        metric: float,
        epoch: int,
        model: nn.Module,
        predictor: nn.Module,
    ) -> bool:
        if self._is_better(metric):
            self.best_metric = metric
            self.best_epoch = epoch
            self.best_model_state = {k: v.detach().cpu().clone()
                                     for k, v in model.state_dict().items()}
            self.best_pred_state = {k: v.detach().cpu().clone()
                                    for k, v in predictor.state_dict().items()}
            self.num_bad_epochs = 0
            return False
        else:
            self.num_bad_epochs += 1
            return self.num_bad_epochs >= self.patience