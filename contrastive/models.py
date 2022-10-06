from typing import List, Optional, Tuple, Union
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from transformers.adapters.modeling import Adapter
from transformers.adapters import (
    BartAdapterModel,
    RobertaAdapterModel,
    BertAdapterModel,
    AdapterConfig,
)
import torch
from torch import nn
from torch.nn import MSELoss


patch_typeguard()


@typechecked
class Eraser(Adapter):
    def __init__(
        self,
        adapter_name,
        input_size,
        down_sample,
        config: AdapterConfig,
    ):
        super().__init__(
            adapter_name,
            input_size,
            down_sample,
            config,
        )
        self.input_size = input_size
        self.adapter_down[0].weight.data.copy_(torch.eye(input_size))
        self.adapter_down[0].bias.data.zero_()
        self.adapter_up.weight.data.copy_(torch.eye(input_size))
        self.adapter_up.bias.data.zero_()
        self.adapter_up.weight.requires_grad_(False)
        self.adapter_up.bias.requires_grad_(False)

    def forward(self, x, residual_input, output_gating=False):
        output = self.adapter_down(x)

        return output, None, None


@typechecked
class AlignmentMixin:
    def __init__(self, config):
        config.hidden_dropout_prob = 0.0
        config.attention_probs_dropout_prob = 0.0
        super().__init__(config)
        self.aligner = nn.Linear(config.hidden_size, config.hidden_size)
        self.new_position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

    @torch.no_grad()
    def produce_original_embeddings(
        self,
        input_ids: TensorType["batch", "seq_len"],
        attention_mask: TensorType["batch", "seq_len"],
        token_type_ids: Optional[TensorType["batch", "seq_len"]] = None,
        position_ids: Optional[TensorType["batch", "seq_len"]] = None,
        head_mask: Optional[TensorType["layers", "heads"]] = None,
    ) -> TensorType["batch", "layers", "hidden_size"]:
        self.train(False)
        if position_ids is None:
            input_shape = input_ids.size()
            seq_length = input_shape[1]
            position_ids = self.base_model.embeddings.position_ids[:, 0:seq_length]
        embedding_output = self.base_model.embeddings.word_embeddings(input_ids)
        self.aligner.weight.data.copy_(torch.eye(self.aligner.in_features))
        self.aligner.bias.zero_()
        self.new_position_embeddings.weight.data.copy_(
            self.base_model.embeddings.position_embeddings.weight.data
        )
        inputs_embeds = (
            self.aligner(embedding_output)
            - self.base_model.embeddings.position_embeddings(position_ids)
            + self.new_position_embeddings(position_ids)
        )
        assert (
            self.base_model.embeddings.position_embeddings(position_ids).sum()
            - self.new_position_embeddings(position_ids).sum()
            == 0
        )
        assert (embedding_output - inputs_embeds).sum() < 0.0001
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        if "hidden_states" in outputs:
            hidden_mat = torch.stack(
                [
                    hidden_state[:, 0, :].squeeze()
                    for hidden_state in outputs.hidden_states[-1:]
                ],
                dim=1,
            )
        else:
            hidden_mat = torch.stack(
                [
                    hidden_state[:, 0, :].squeeze()
                    for hidden_state in outputs.encoder_hidden_states[-1:]
                ],
                dim=1,
            )
        self.train(True)
        return hidden_mat

    def _euclidean_distance(
        self,
        x: TensorType["batch", "hidden_size"],
        y: TensorType["batch", "hidden_size"],
    ) -> TensorType["batch"]:
        distance = torch.sum((x - y) ** 2, 1).squeeze()
        return distance.sqrt()

    def forward(
        self,
        input_ids: TensorType["batch", "seq_len"],
        attention_mask: TensorType["batch", "seq_len"],
        original_embedding: TensorType["batch", "layers", "hidden_size"],
        token_type_ids: Optional[TensorType["batch", "seq_len"]] = None,
        position_ids: Optional[TensorType["batch", "seq_len"]] = None,
        head_mask: Optional[TensorType["layers", "heads"]] = None,
    ) -> Tuple[torch.Tensor]:
        embedding_output = self.base_model.embeddings.word_embeddings(input_ids)
        if position_ids is None:
            input_shape = input_ids.size()
            seq_length = input_shape[1]
            position_ids = self.base_model.embeddings.position_ids[:, 0:seq_length]
        inputs_embeds = (
            self.aligner(embedding_output)
            - self.base_model.embeddings.position_embeddings(position_ids)
            + self.new_position_embeddings(position_ids)
        )
        outputs = super().forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        if "hidden_states" in outputs:
            hidden_mat = torch.stack(
                [
                    hidden_state[:, 0, :].squeeze()
                    for hidden_state in outputs.hidden_states[-1:]
                ],
                dim=1,
            )
        else:
            hidden_mat = torch.stack(
                [
                    hidden_state[:, 0, :].squeeze()
                    for hidden_state in outputs.encoder_hidden_states[-1:]
                ],
                dim=1,
            )
        loss = torch.linalg.norm(
            hidden_mat.flatten(start_dim=1) - original_embedding.flatten(start_dim=1),
            ord=2,
            dim=1,
        )
        loss = torch.mean(loss)
        loss = self._euclidean_distance(
            hidden_mat.flatten(start_dim=1),
            original_embedding.flatten(start_dim=1),
        )
        loss = torch.mean(loss)
        return (loss,)


@typechecked
class BartAdapterModelForAlignment(AlignmentMixin, BartAdapterModel):
    def __init__(self, config):
        config.dropout = 0.0
        config.activation_dropout = 0.0
        config.attention_dropout = 0.0
        config.classifier_dropout = 0.0
        super().__init__(config)


@typechecked
class RobertaAdapterModelForAlignment(AlignmentMixin, RobertaAdapterModel):
    def __init__(self, config):
        config.hidden_dropout_prob = 0.0
        config.attention_probs_dropout_prob = 0.0
        super().__init__(config)


@typechecked
class BertAdapterModelForAlignment(AlignmentMixin, BertAdapterModel):
    def __init__(self, config):
        config.hidden_dropout_prob = 0.0
        config.attention_probs_dropout_prob = 0.0
        super().__init__(config)
