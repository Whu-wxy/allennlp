from typing import List

from overrides import overrides
import torch
from torch.nn import Dropout

from allennlp.modules.feedforward import FeedForward
from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.seq2seq_encoders.multi_head_coattention import MultiHeadCoAttention
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn.activations import Activation
from allennlp.nn.util import add_positional_features


@Seq2SeqEncoder.register("stacked_coattention")
class StackedCoattentionEncoder(Seq2SeqEncoder):
    # pylint: disable=line-too-long
    """
    Implements a stacked self-attention encoder similar to the Transformer
    architecture in `Attention is all you Need
    <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

    This encoder combines 3 layers in a 'block':

    1. A 2 layer FeedForward network.
    2. Multi-headed coattention, which uses 2 learnt linear projections
       to perform a dot-product similarity between every pair of elements
       scaled by the square root of the sequence length.
    3. Layer Normalisation.

    These are then stacked into ``num_layers`` layers.

    Parameters
    ----------
    input_dim : ``int``, required.
        The input dimension of the encoder.
    feedforward_hidden_dim : ``int``, required.
        The middle dimension of the FeedForward network. The input and output
        dimensions are fixed to ensure sizes match up for the self attention layers.
    num_layers : ``int``, required.
        The number of stacked self attention -> feedfoward -> layer normalisation blocks.
    num_attention_heads : ``int``, required.
        The number of attention heads to use per layer.
    use_positional_encoding: ``bool``, optional, (default = True)
        Whether to add sinusoidal frequencies to the input tensor. This is strongly recommended,
        as without this feature, the self attention layers have no idea of absolute or relative
        position (as they are just computing pairwise similarity between vectors of elements),
        which can be important features for many tasks.
    dropout_prob : ``float``, optional, (default = 0.1)
        The dropout probability for the feedforward network.
    residual_dropout_prob : ``float``, optional, (default = 0.2)
        The dropout probability for the residual connections.
    attention_dropout_prob : ``float``, optional, (default = 0.1)
        The dropout probability for the attention distributions in each attention layer.
    """
    def __init__(self,
                 input_dim: int,
                 feedforward_hidden_dim: int,
                 num_layers: int,
                 num_attention_heads: int,
                 use_positional_encoding: bool = True,
                 dropout_prob: float = 0.1,
                 residual_dropout_prob: float = 0.2,
                 attention_dropout_prob: float = 0.1) -> None:
        super(StackedCoattentionEncoder, self).__init__()

        self._use_positional_encoding = use_positional_encoding
        self._attention_layers: List[MultiHeadCoAttention] = []
        self.p_feedfoward_layers: List[FeedForward] = []
        self.q_feedfoward_layers: List[FeedForward] = []
        self.q_feed_forward_layer_norm_layers: List[LayerNorm] = []
        self.p_feed_forward_layer_norm_layers: List[LayerNorm] = []
        self._layer_norm_layers: List[LayerNorm] = []

        for i in range(num_layers):
            p_feedfoward = FeedForward(input_dim * 4,
                                     activations=[Activation.by_name('relu')(),
                                                  Activation.by_name('linear')()],
                                     hidden_dims=[feedforward_hidden_dim, input_dim],
                                     num_layers=2,
                                     dropout=dropout_prob)
            q_feedfoward = FeedForward(input_dim,
                                     activations=[Activation.by_name('relu')(),
                                                  Activation.by_name('linear')()],
                                     hidden_dims=[feedforward_hidden_dim, input_dim],
                                     num_layers=2,
                                     dropout=dropout_prob)

            self.add_module(f"p_feedforward_{i}", p_feedfoward)
            self.add_module(f"q_feedforward_{i}", q_feedfoward)
            self.p_feedfoward_layers.append(p_feedfoward)
            self.q_feedfoward_layers.append(q_feedfoward)

            p_feedforward_layer_norm = LayerNorm(p_feedfoward.get_output_dim())
            q_feedforward_layer_norm = LayerNorm(q_feedfoward.get_output_dim())
            self.add_module(f"p_feedforward_layer_norm_{i}", p_feedforward_layer_norm)
            self.add_module(f"q_feedforward_layer_norm_{i}", q_feedforward_layer_norm)
            self.p_feed_forward_layer_norm_layers.append(p_feedforward_layer_norm)
            self.q_feed_forward_layer_norm_layers.append(q_feedforward_layer_norm)

            coattention = MultiHeadCoAttention(num_heads=num_attention_heads,
                                                    input_dim=input_dim,
                                                    attention_dropout_prob=attention_dropout_prob)
            self.add_module(f"coattention_{i}", coattention)
            self._attention_layers.append(coattention)

            layer_norm = LayerNorm(coattention.get_output_dim())
            self.add_module(f"layer_norm_{i}", layer_norm)
            self._layer_norm_layers.append(layer_norm)

        self.dropout = Dropout(residual_dropout_prob)
        self._input_dim = input_dim
        self._output_dim = input_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return True

    @overrides
    def forward(self, passage_tensor: torch.Tensor,
                question_tensor:torch.Tensor,
                passage_mask: torch.LongTensor = None,
                question_mask: torch.LongTensor = None): # pylint: disable=arguments-differ

        if self._use_positional_encoding:
            passage_tensor = add_positional_features(passage_tensor)
            question_tensor = add_positional_features(question_tensor)

        cached_passage = passage_tensor
        cached_question = question_tensor

        for (attention,
             p_feedforward,
             q_feedforward,
             p_feedforward_layer_norm,
             q_feedforward_layer_norm,
             layer_norm) in zip(self._attention_layers,
                                self.p_feedfoward_layers,
                                self.q_feedfoward_layers,
                                self.p_feed_forward_layer_norm_layers,
                                self.q_feed_forward_layer_norm_layers,
                                self._layer_norm_layers):


            # shape (batch_size, sequence_length, input_dim), (batch_size, sequence_length, input_dim)
            passage_question_vectors, passage_passage_vectors = attention(cached_passage, cached_question, passage_mask, question_mask)
            # Add & LayerNorm
            passage_passage_vectors = layer_norm(self.dropout(passage_passage_vectors) + cached_passage)
            passage_question_vectors = layer_norm(self.dropout(passage_question_vectors) + cached_passage)

            # shape (batch_size, sequence_length, input_dim)
            cached_passage = passage_passage_vectors

            # shape (batch_size, timesteps, input_size*4)
            merged_passage_attention_vectors = self.dropout(
                torch.cat([passage_tensor, passage_question_vectors,
                           passage_tensor * passage_question_vectors,
                           passage_tensor * passage_passage_vectors],
                          dim=-1)
            )
            # feedforward
            p_feedforward_output = p_feedforward(merged_passage_attention_vectors)
            p_feedforward_output = self.dropout(p_feedforward_output)
            q_feedforward_output = q_feedforward(cached_question)
            q_feedforward_output = self.dropout(q_feedforward_output)

            # Add & LayerNorm
            p_feedforward_output = p_feedforward_layer_norm(p_feedforward_output + cached_passage)
            q_feedforward_output = q_feedforward_layer_norm(q_feedforward_output + cached_question)

            cached_passage = p_feedforward_output
            cached_question = q_feedforward_output

        p_feedforward_output = self.dropout(cached_passage)
        q_feedforward_output = self.dropout(cached_question)

        # shape: (batch_size, sequence_length, input_dim)
        return p_feedforward_output, q_feedforward_output
