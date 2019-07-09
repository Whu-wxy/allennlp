import torch.nn.functional as F
import torch.nn as nn
import torch

from typing import Optional, Tuple

from allennlp.modules.locked_dropout import LockedDropout
from allennlp.modules.layer_norm import LayerNorm
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence

class LinearDropConnect(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dropout=0.):
        super(LinearDropConnect, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        self.dropout = dropout

    def sample_mask(self):
        if self.dropout == 0.:
            self._weight = self.weight
        else:
            mask = self.weight.new_empty(
                self.weight.size(),
                dtype=torch.uint8
            )
            mask.bernoulli_(self.dropout)
            self._weight = self.weight.masked_fill(mask, 0.)

    def forward(self, input, sample_mask=False):
        if self.training:
            if sample_mask:
                self.sample_mask()
            return F.linear(input, self._weight, self.bias)
        else:
            return F.linear(input, self.weight * (1 - self.dropout),
                            self.bias)


def cumsoftmax(x, dim=-1):
    softmax = torch.nn.Softmax(dim=dim)
    return torch.cumsum(softmax(x), dim=dim)


class ONLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, chunk_size, dropconnect=0.):
        super(ONLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.n_chunk = int(hidden_size / chunk_size)

        self.ih = nn.Sequential(
            nn.Linear(input_size, 4 * hidden_size + self.n_chunk * 2, bias=True),
            # LayerNorm(3 * hidden_size)
        )
        self.hh = LinearDropConnect(hidden_size, hidden_size*4+self.n_chunk*2, bias=True, dropout=dropconnect)

        # self.c_norm = LayerNorm(hidden_size)

        self.drop_weight_modules = [self.hh]

        self.reset_parameters()

    def forward(self, input, hx, cx,
                transformed_input=None):

        if transformed_input is None:
            transformed_input = self.ih(input)

        gates = transformed_input + self.hh(hx)
        cingate, cforgetgate = gates[:, :self.n_chunk*2].chunk(2, 1)
        outgate, cell, ingate, forgetgate = gates[:,self.n_chunk*2:].view(-1, self.n_chunk*4, self.chunk_size).chunk(4,1)

        cingate = 1. - cumsoftmax(cingate)
        cforgetgate = cumsoftmax(cforgetgate)

        cingate = cingate[:, :, None]
        cforgetgate = cforgetgate[:, :, None]

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cell = torch.tanh(cell)
        outgate = torch.sigmoid(outgate)

        # cy = cforgetgate * forgetgate * cx + cingate * ingate * cell

        overlap = cforgetgate * cingate
        forgetgate = forgetgate * overlap + (cforgetgate - overlap)
        ingate = ingate * overlap + (cingate - overlap)
        cy = forgetgate * cx + ingate * cell

        # hy = outgate * F.tanh(self.c_norm(cy))
        hy = outgate * torch.tanh(cy)
        return hy.view(-1, self.hidden_size), cy

    def reset_parameters(self):
        pass

    def sample_masks(self):
        for m in self.drop_weight_modules:
            m.sample_mask()


class ONLSTMStack(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 chunk_size,
                 dropout=0.,
                 dropconnect=0.) -> None:
        super(ONLSTMStack, self).__init__()

        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.n_chunk = int(hidden_size / chunk_size)

        self.cells = nn.ModuleList([])
        lstm_input_size = input_size
        for layer_index in range(num_layers):
            layer = ONLSTMCell(lstm_input_size, hidden_size, chunk_size, dropconnect=dropconnect)
            lstm_input_size = hidden_size
            self.add_module('layer_{}'.format(layer_index), layer)
            self.cells.append(layer)

        self.lockdrop = LockedDropout()
        self.dropout = dropout

    def forward(self,  # pylint: disable=arguments-differ
                sequence_tensor: PackedSequence,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) :

        if not isinstance(inputs, PackedSequence):
            raise ConfigurationError('inputs must be PackedSequence but got %s' % (type(inputs)))

        sequence_tensor, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        batch_size = sequence_tensor.size()[0]
        length = sequence_tensor.size()[1]

        hidden = torch.tensor([])
        cell = torch.tensor([])
        #初始化所有层隐藏状态
        if initial_state is None:
            hidden = sequence_tensor.new_zeros(self.num_layers, batch_size, self.hidden_size)
            cell = sequence_tensor.new_zeros(self.num_layers, batch_size, self.n_chunk, self.chunk_size)
        else:
            hidden = initial_state[0].squeeze(0)
            cell = initial_state[1].squeeze(0)

        if self.training:
            for c in self.cells:
                c.sample_masks()

        final_hidden = []
        final_cell = []

        for l in range(len(self.cells)):
            curr_layer = [None] * length
            t_input = self.cells[l].ih(sequence_tensor)
            hx = hidden[l].squeeze(0)
            cx = cell[l].squeeze(0)

            for t in range(length):
                hidden, cell = self.cells[l](None, hx, cx, transformed_input=t_input[:, t])
                # length, dim
                hx, cx = hidden, cell  # overwritten every timestep
                curr_layer[t] = hidden

            final_hidden.append(hx)
            final_cell.append(cx)
            # batch, length, dim
            sequence_tensor = torch.stack(curr_layer, dim=1)
            if l < len(self.cells) - 1:
                sequence_tensor = self.lockdrop(sequence_tensor, self.dropout)  #每一层LSTM后加lockdrop


        output = pack_padded_sequence(sequence_tensor, batch_lengths, batch_first=True)
        final_state = (torch.stack(final_hidden), torch.stack(final_cell))

        return sequence_tensor, final_state

if __name__ == "__main__":
    x = torch.ones(2, 3, 6)
    x.data.normal_()
    mylstm = ONLSTMStack(6, 6, 2, chunk_size=3)
    mylstm.eval()
    print(mylstm(x)[0])