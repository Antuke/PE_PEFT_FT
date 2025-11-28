"""Task class definition"""

from dataclasses import dataclass
from typing import List, Type
import torch.nn as nn


@dataclass
class Task:
    """Encapsulates all configuration for a single task."""

    name: str
    class_labels: List[str]
    criterion: Type[nn.Module]
    weight: float = 1.0
    use_weighted_loss: bool = False

    @property
    def num_classes(self) -> int:
        return len(self.class_labels)
