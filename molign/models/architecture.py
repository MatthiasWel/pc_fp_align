from collections import OrderedDict

import torch
from torch.nn import Identity, LayerNorm, Linear, ReLU, Sequential, Sigmoid


class BaseMLP(torch.nn.Module):
    def __init__(self, model_config):
        super(BaseMLP, self).__init__()
        self.decoder_input = model_config.get(
            "decoder_input", model_config.get("hidden_channels", 64)
        )
        self.hidden_dimensions = model_config.get("hidden_dimensions", 64)
        self.batch_norm = (
            LayerNorm if model_config.get("batch_norm", False) else Identity
        )
        self.dropout_p = model_config.get("dropout_p", 0.0)
        self.activation = model_config.get("activation", ReLU)
        self.num_linear_layers = model_config.get("num_linear_layers", 2)

        self.MLP = self._MLP()
        self.last_layer = self._last_layer()

    def _MLP(self):
        if self.num_linear_layers == 0:
            return Identity()
        return Sequential(
            OrderedDict(
                [
                    ("lin_in", Linear(self.decoder_input, self.hidden_dimensions)),
                    ("norm_in", self.batch_norm(self.hidden_dimensions)),
                    ("act_in", self.activation()),
                ]
                + [self._MLP_block(i) for i in range((self.num_linear_layers - 2) * 3)]
            )
        )

    def _MLP_block(self, i):
        func_type = i % 3
        block_nr = i // 3 + 1
        if func_type == 0:
            return (
                f"lin{block_nr}",
                Linear(self.hidden_dimensions, self.hidden_dimensions),
            )
        if func_type == 1:
            return (f"norm{block_nr}", self.batch_norm(self.hidden_dimensions))
        if func_type == 2:
            return (f"act{block_nr}", self.activation())
        assert False, "Something went wrong in _MLP_block with the func_type"

    def _last_layer(self):
        raise NotImplementedError

    def forward(self, x):
        x = self.MLP(x)
        return self.last_layer(x)

    def embedding(self, x):
        return self.MLP(x)


class BinaryClassificationMLP(BaseMLP):
    def __init__(self, model_config):
        super(BinaryClassificationMLP, self).__init__(model_config)

    def _last_layer(self):
        return Sequential(
            OrderedDict(
                [
                    ("lin_out", Linear(self.hidden_dimensions, 1)),
                    ("sigmoid_out", Sigmoid()),
                ]
            )
        )
