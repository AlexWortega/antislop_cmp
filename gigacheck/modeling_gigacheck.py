from transformers import AutoTokenizer
from transformers.modeling_outputs import ModelOutput
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import torch

from gigacheck.model.mistral_ai_detector import MistralAIDetectorForSequenceClassification
from gigacheck.model.src.interval_detector.span_utils import span_cxw_to_xx

from .configuration_gigacheck import GigaCheckConfig


@dataclass
class GigaCheckOutput(ModelOutput):
    """
    Output type for GigaCheck model.

    Args:
        pred_label_ids (torch.Tensor): [Batch] Indices of the predicted classes (Human/AI/Mixed).
        classification_head_probs (torch.Tensor): [Batch, Num_Classes] Softmax probabilities.
    """
    pred_label_ids: Optional[torch.Tensor] = None
    classification_head_probs: Optional[torch.Tensor] = None


class GigaCheckForSequenceClassification(MistralAIDetectorForSequenceClassification):
    config_class = GigaCheckConfig

    def __init__(self, config: GigaCheckConfig):
        super().__init__(
            config,
            with_detr = False,
            detr_config = None,
            ce_weights = None,
            freeze_backbone = False,
            id2label = config.id2label,
        )
        self.trained_classification_head = True
        self._max_len = self.config.max_length
        self.tokenizer = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):  # type: ignore
        """Loads a pretrained GigaCheck model from a local path or the Hugging Face Hub.

            Args:
                pretrained_model_name_or_path (str): The name or path of the pretrained model.
                model_args: Additional positional arguments passed to parent class.
                kwargs: Additional keyword arguments passed to parent class.

            Returns:
                GigaCheckForSequenceClassification: The initialized model with loaded weights and initialized tokenizer.
        """
        # set model weights
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )

        if model.config.to_dict().get("trained_classification_head", True) is False:
            # when only detr was trained
            model.trained_classification_head = False

        model.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        # Ensure pad token exists
        model.config.pad_token_id = model.tokenizer.pad_token_id \
            if model.tokenizer.pad_token_id is not None else model.tokenizer.unk_token_id
        if model.tokenizer.pad_token_id is None:
            model.tokenizer.pad_token_id = model.tokenizer.unk_token_id

        model.config.bos_token_id = model.tokenizer.bos_token_id
        model.config.eos_token_id = model.tokenizer.eos_token_id
        model.config.unk_token_id = model.tokenizer.unk_token_id

        return model

    def _get_inputs(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Tokenizes a batch of texts handling specific truncation logic to preserve exact text length mapping.
        """
        assert self._max_len is not None and self.tokenizer is not None, "Model must be initialized"

        # 1. Tokenize all texts without special tokens/padding first
        raw_encodings = self.tokenizer(texts, add_special_tokens=False)

        batch_features = []  # List of dicts for tokenizer.pad
        text_lens = []

        content_max_len = self._max_len - 2
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        for i, tokens in enumerate(raw_encodings.input_ids):
            if len(tokens) > content_max_len:
                tokens = tokens[:content_max_len]
                # Convert back to string to get the exact character length of the truncated part
                cur_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                text_len = len(cur_text)
            else:
                # If no truncation, use the original text length
                text_len = len(texts[i])

            # Construct final token sequence: [BOS] + tokens + [EOS]
            final_tokens = [bos_id] + tokens + [eos_id]

            # Append as dictionary for tokenizer.pad
            batch_features.append({"input_ids": final_tokens})
            text_lens.append(text_len)

        # 2. Pad using tokenizer.pad
        padded_output = self.tokenizer.pad(
            batch_features,
            padding=True,
            return_tensors="pt"
        )

        input_ids = padded_output["input_ids"].to(self.device)
        attention_mask = padded_output["attention_mask"].to(self.device)

        return input_ids, attention_mask, text_lens

    def forward(
        self,
        text: Union[str, List[str]],
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, GigaCheckOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if isinstance(text, str):
            text = [text]

        input_ids, attention_mask, text_lens = self._get_inputs(text)

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            return_detr_output=self.config.with_detr,
        )

        # 1. Classification Head Processing
        logits = output.logits

        # logits: [Batch, NumClasses]
        probs = logits.to(torch.float32).softmax(dim=-1)
        pred_label_ids = torch.argmax(probs, dim=-1)  # [Batch]
        classification_head_probs = probs  # [Batch, NumClasses]

        if not return_dict:
            return (pred_label_ids, classification_head_probs)

        return GigaCheckOutput(
            pred_label_ids=pred_label_ids,
            classification_head_probs=classification_head_probs,
        )


def to_absolute(pred_spans: torch.Tensor, text_len: int) -> torch.Tensor:
    spans = span_cxw_to_xx(pred_spans) * text_len
    return torch.clamp(spans, 0, text_len)
