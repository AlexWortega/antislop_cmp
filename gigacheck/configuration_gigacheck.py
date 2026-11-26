from typing import Dict, Optional, Any
from transformers import MistralConfig


class GigaCheckConfig(MistralConfig):
    def __init__(
        self,
        with_detr: bool = False,
        detr_config: Optional[Dict[str, Any]] = None,
        freeze_backbone: bool = False,
        id2label: Dict[int, str] = None,
        num_labels: int = 2,
        max_length: int = 1024,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.with_detr = with_detr
        self.detr_config = detr_config
        self.freeze_backbone = freeze_backbone
        self.id2label = id2label
        self.num_labels = num_labels
        self.max_length = max_length

        if self.id2label:
            self.id2label = {int(k): v for k, v in self.id2label.items()}
