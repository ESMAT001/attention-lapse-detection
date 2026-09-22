import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from attention_lapse_detection.training.metrics import ap_disengaged
from attention_lapse_detection.utils.constants import PATIENCE
from attention_lapse_detection.utils.numeric import round4


class Trainer:
    """Trains one model: epoch loop, validation, early stopping on val AP.

    Keeps the best epoch's weights and the validation outputs it produced
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer

        # Validation outputs of the kept epoch, populated by fit()
        self.val_y_true = np.empty(0, dtype=np.int64)
        self.val_y_pred = np.empty(0, dtype=np.int64)
        self.val_probs = np.empty((0, 0), dtype=np.float32)

        self.best_val_ap = float("nan")
        self.best_epoch = 0
        self.epochs_run = 0
        # One dict per epoch
        self.history: list[dict[str, float]] = []

    def train_one_epoch(self) -> float:
        """One pass over the training data, returns the average batch loss"""

        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(X_batch), y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        """Run the model on the validation set.

        Returns (avg loss, accuracy, true labels, predictions, probabilities).
        """
        self.model.eval()

        total_loss = 0.0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_probs: list[torch.Tensor] = []

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.model(X_batch)

                total_loss += self.criterion(outputs, y_batch).item()

                all_probs.append(torch.softmax(outputs, dim=1).cpu())
                all_preds.append(outputs.argmax(dim=1).cpu())
                all_labels.append(y_batch.long().cpu())

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        probs = torch.cat(all_probs).numpy()
        accuracy = float((y_pred == y_true).mean())

        return total_loss / len(self.val_loader), accuracy, y_true, y_pred, probs

    def fit(
        self,
        epochs: int,
        early_stopping: bool = False,
        patience: int = PATIENCE,
    ) -> None:
        """Train for up to 50 epochs, optionally stopping early on val AP."""

        best_ap = -np.inf
        best_epoch = 0
        epochs_since_improvement = 0
        best_state: dict[str, torch.Tensor] | None = None
        last_epoch = 0

        for epoch in range(1, epochs + 1):
            last_epoch = epoch
            train_loss = self.train_one_epoch()
            val_loss, val_accuracy, y_true, y_pred, probs = self.evaluate()

            self.val_y_true = y_true
            self.val_y_pred = y_pred
            self.val_probs = probs

            log = (
                f"Epoch {epoch}/{epochs} | "
                f"train_loss: {round4(train_loss)} | "
                f"val_loss: {round4(val_loss)} | "
                f"val_acc: {round4(val_accuracy)}"
            )

            epoch_record: dict[str, float] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_accuracy,
                "lr": self.optimizer.param_groups[0]["lr"],
            }

            if early_stopping:
                val_ap = ap_disengaged(y_true, probs)
                log += f" | val_ap: {round4(val_ap)}"
                epoch_record["val_ap"] = val_ap

                if val_ap > best_ap:
                    best_ap = val_ap
                    best_epoch = epoch
                    epochs_since_improvement = 0
                    best_state = copy.deepcopy(self.model.state_dict())
                else:
                    epochs_since_improvement += 1

            self.history.append(epoch_record)
            print(log)

            if early_stopping and epochs_since_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch}: "
                    f"no val AP improvement for {patience} epochs."
                )
                break

        self.epochs_run = last_epoch

        if early_stopping:
            self.best_val_ap = best_ap
            self.best_epoch = best_epoch
            if best_state is not None:
                # Roll back to the best epoch's weights and re-score
                self.model.load_state_dict(best_state)
                _, _, y_true, y_pred, probs = self.evaluate()
                self.val_y_true = y_true
                self.val_y_pred = y_pred
                self.val_probs = probs
            print(
                f"Best epoch: {best_epoch} | "
                f"best val AP: {round4(best_ap)} | "
                f"epochs run: {last_epoch}"
            )

        print("\nValidation classification report:")
        print(
            classification_report(
                self.val_y_true,
                self.val_y_pred,
                target_names=["disengaged", "engaged"],
                digits=4,
                zero_division=0,
            )
        )
        print("Confusion matrix (rows = true, cols = predicted):")
        print(confusion_matrix(self.val_y_true, self.val_y_pred))
