from typing import List, Optional, Tuple, Union
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from transformers.adapters.modeling import Adapter
from transformers.adapters import RobertaAdapterModel, BertAdapterModel, AdapterConfig
from scipy.linalg import orthogonal_procrustes
import torch
from torch import nn
from torch.nn import MSELoss


patch_typeguard()


@typechecked
class ProcrustesMixin:
    def _euclidean_distance(
        self,
        x: TensorType["batch", "hidden_size"],
        y: TensorType["batch", "hidden_size"],
    ) -> TensorType["batch"]:
        distance = torch.sum((x - y) ** 2, 1).squeeze()
        return distance.sqrt()

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

        i = 12
        hidden_mat = torch.stack(
            [
                hidden_state[:, 0, :]
                for hidden_state in outputs.hidden_states[i : i + 1]
            ],
            dim=1,
        )
        self.train(True)
        return hidden_mat

    def compute_shift(self, A):
        mu = A.mean(axis=0)
        return mu

    @torch.no_grad()
    def align(
        self,
        original_embedding: TensorType["batch", "hidden_size"],
        transformed_embedding: TensorType["batch", "hidden_size"],
        solve=True,
    ) -> Tuple[torch.Tensor]:
        self.train(False)
        orig = original_embedding.flatten(start_dim=1)
        transformed = transformed_embedding.flatten(start_dim=1)

        orig_loss = self._euclidean_distance(transformed, orig)
        orig_loss = torch.mean(orig_loss)
        print(orig_loss)

        transformed = torch.cat(
            (
                torch.ones(transformed.shape[0], device=transformed.device).unsqueeze(
                    dim=1
                ),
                transformed,
            ),
            dim=1,
        )

        if solve:
            self.rotate = torch.linalg.lstsq(transformed, orig).solution

        loss = self._euclidean_distance(
            torch.matmul(transformed, self.rotate),
            orig,
        )
        loss = torch.mean(loss)
        print(loss)
        return (loss,)

    # @torch.no_grad()
    # def align(
    #     self,
    #     original_embedding: TensorType["batch", "hidden_size"],
    #     transformed_embedding: TensorType["batch", "hidden_size"],
    #     solve=True,
    # ) -> Tuple[torch.Tensor]:
    #     self.train(False)
    #     orig = original_embedding.flatten(start_dim=1)
    #     transformed = transformed_embedding.flatten(start_dim=1)

    #     # orig = torch.tensor([[-3, 3], [-2, 3], [-2, 2], [-3, 2.0]]).cuda()
    #     # transformed = torch.tensor([[3, 2], [1, 0], [3, -2], [5, 0.0]]).cuda()

    #     orig_loss = self._euclidean_distance(transformed, orig)
    #     orig_loss = torch.mean(orig_loss)
    #     print(orig_loss)

    #     if solve:
    #         self.orig_shift = self.compute_shift(orig)
    #         self.transformed_shift = self.compute_shift(transformed)
    #         R, s = orthogonal_procrustes(
    #             (transformed - self.transformed_shift).cpu().numpy(),
    #             (orig + self.orig_shift).cpu().numpy(),
    #         )
    #         self.rotate = torch.tensor(R).cuda()
    #         self.scale = s / (transformed - self.transformed_shift).norm().square()

    #     loss = self._euclidean_distance(
    #         self.scale * torch.matmul(transformed - self.transformed_shift, self.rotate)
    #         + self.orig_shift,
    #         orig,
    #     )
    #     loss = torch.mean(loss)
    #     print(loss)

    #     return (loss,)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if self.bert:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            outputs = super().forward(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        sequence_output = outputs[0]

        outputs[0] = torch.matmul(sequence_output, self.rotate)
        return
