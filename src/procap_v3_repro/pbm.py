"""Pro-Cap V3 — PromptHateModel (port of upstream pbm.py).

Source: `external_repos/procap/codes/scr/pbm.py:5-55` (verbatim
structure, same RoBERTa-large `*ForMaskedLM`, same logit slice over
`label_word_list`). The only changes are:

  * Optional pluggable `model_name` (still defaults to roberta-large)
  * Tokenizer is created **inside** the class (matches upstream); the
    caller doesn't need to manage it separately.

Upstream `forward(all_texts)` takes a list of strings and returns
logits of shape `[B, num_labels]` over the label word ids.
"""

import torch
import torch.nn as nn


class PromptHateModel(nn.Module):
    def __init__(
        self,
        label_words=("good", "bad"),
        max_length=320,
        model_name="roberta-large",
    ):
        super().__init__()
        from transformers import RobertaForMaskedLM, RobertaTokenizer

        self.model_name = model_name
        self.roberta = RobertaForMaskedLM.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        self.mask_token_id = self.tokenizer.mask_token_id
        self.label_word_list = []
        for word in label_words:
            tok_id = self.tokenizer._convert_token_to_id(
                self.tokenizer.tokenize(" " + word)[0]
            )
            self.label_word_list.append(tok_id)

    def forward_single_cap(self, tokens, attention_mask, mask_pos):
        """Upstream `pbm.py:19-33`, verbatim."""
        batch_size = tokens.size(0)
        mask_pos = mask_pos.squeeze()

        out = self.roberta(tokens, attention_mask)
        prediction_mask_scores = out.logits[
            torch.arange(batch_size), mask_pos
        ]
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(
                prediction_mask_scores[
                    :, self.label_word_list[label_id]
                ].unsqueeze(-1)
            )
        logits = torch.cat(logits, -1)
        return logits

    def generate_input_tokens(self, sents, max_length=None):
        """Upstream `pbm.py:35-43`, verbatim. Returns
        `(tokens, attention_mask, mask_pos)`.
        """
        if max_length is None:
            max_length = self.max_length
        token_info = self.tokenizer(
            sents,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = token_info.input_ids
        attention_mask = token_info.attention_mask
        mask_pos = [
            t.numpy().tolist().index(self.mask_token_id) for t in tokens
        ]
        mask_pos = torch.LongTensor(mask_pos)
        return tokens, attention_mask, mask_pos

    def forward(self, all_texts):
        """Upstream `pbm.py:45-52`, verbatim."""
        tokens, attention_mask, mask_pos = self.generate_input_tokens(
            all_texts, self.max_length
        )
        device = self.roberta.device
        logits = self.forward_single_cap(
            tokens.to(device),
            attention_mask.to(device),
            mask_pos.to(device),
        )
        return logits


def build_baseline(label_words, max_length, model_name="roberta-large"):
    """Upstream `pbm.py:54-55`."""
    return PromptHateModel(label_words, max_length, model_name=model_name)
