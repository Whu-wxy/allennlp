from overrides import overrides
import torch
from torch.nn import Dropout, Linear
from torch.nn import LayerNorm
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.activations import Activation
from allennlp.nn.util import masked_softmax, weighted_sum
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules import Highway
from torch.nn.functional import relu
from allennlp.nn.util import add_positional_features

@Seq2SeqEncoder.register("multi_head_coattention44")
class MultiHeadCoAttention44(Seq2SeqEncoder):
    # pylint: disable=line-too-long
    """
    This class implements the key-value scaled dot product attention mechanism
    detailed in the paper `Attention is all you Need
    <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

    The attention mechanism is a weighted sum of a projection V of the inputs, with respect
    to the scaled, normalised dot product of Q and K, which are also both linear projections
    of the input. This procedure is repeated for each attention head, using different parameters.

    Parameters
    ----------
    num_heads : ``int``, required.
        The number of attention heads to use.
    input_dim : ``int``, required.
        The size of the last dimension of the input tensor.
    output_projection_dim : ``int``, optional (default = None)
        The dimensionality of the final output projection. If this is not passed
        explicitly, the projection has size `input_size`.
    attention_dropout_prob : ``float``, optional (default = 0.1).
        The dropout probability applied to the normalised attention
        distributions.
    """
    def __init__(self,
                 num_blocks:int,
                 num_heads: int,
                 input_dim: int,
                 attention_dropout_prob: float = 0.1) -> None:
        super(MultiHeadCoAttention44, self).__init__()

        self._num_heads = num_heads
        self._input_dim = input_dim
        self._output_dim = input_dim

        self._coattention_blocks = []
        for block_index in range(num_blocks):
            coattention_block = MultiHeadCoAttention_block44(num_heads,
                                                             input_dim,
                                                             attention_dropout_prob)
            self.add_module(f"coattention_block_{block_index}", coattention_block)
            self._coattention_blocks.append(coattention_block)


    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def is_bidirectional(self) -> bool:
        return False

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                passage_tensor: torch.Tensor,
                question_tensor:torch.Tensor,
                passage_mask: torch.LongTensor = None,
                question_mask: torch.LongTensor = None) -> torch.Tensor:  # pylint: disable=arguments-differ

        passage = passage_tensor
        for coattention_block in self._coattention_blocks:
            passage = coattention_block(passage, question_tensor, passage_mask, question_mask)
        return passage



@Seq2SeqEncoder.register("multi_head_coattention_block44")
class MultiHeadCoAttention_block44(Seq2SeqEncoder):
    # pylint: disable=line-too-long
    """
    This class implements the key-value scaled dot product attention mechanism
    detailed in the paper `Attention is all you Need
    <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

    The attention mechanism is a weighted sum of a projection V of the inputs, with respect
    to the scaled, normalised dot product of Q and K, which are also both linear projections
    of the input. This procedure is repeated for each attention head, using different parameters.

    Parameters
    ----------
    num_heads : ``int``, required.
        The number of attention heads to use.
    input_dim : ``int``, required.
        The size of the last dimension of the input tensor.
    output_projection_dim : ``int``, optional (default = None)
        The dimensionality of the final output projection. If this is not passed
        explicitly, the projection has size `input_size`.
    attention_dropout_prob : ``float``, optional (default = 0.1).
        The dropout probability applied to the normalised attention
        distributions.
    """
    def __init__(self,
                 num_heads: int,
                 input_dim: int,
                 use_positional_encoding: bool = True,
                 attention_dropout_prob: float = 0.1) -> None:
        super(MultiHeadCoAttention_block44, self).__init__()

        self._num_heads = num_heads
        self._input_dim = input_dim
        self._output_dim = input_dim
        self._use_positional_encoding = use_positional_encoding

        if input_dim % num_heads != 0:
            raise ValueError(f"Key size ({input_dim}) must be divisible by the number of "
                             f"attention heads ({num_heads}).")

        #self._combined_projection = Linear(input_dim, 2 * input_dim + values_dim)
        self._combined_projection = Linear(input_dim, 2 * input_dim)  # Query & Value

        self._output_projection = FeedForward(input_dim*4,
                                 activations=[Activation.by_name('relu')(), Activation.by_name('linear')()],
                                 hidden_dims=[input_dim*4, input_dim], num_layers=2, dropout=attention_dropout_prob)

        self._norm_layer = LayerNorm(input_dim)
        self._merged_norm_layer = LayerNorm(input_dim*4)
        self._out_norm_layer = LayerNorm(input_dim)

        self._scale = (input_dim // num_heads) ** 0.5
        self.dropout = Dropout(attention_dropout_prob)


    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return True

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                passage_tensor: torch.Tensor,
                question_tensor:torch.Tensor,
                passage_mask: torch.LongTensor = None,
                question_mask: torch.LongTensor = None) -> torch.FloatTensor:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).

        Returns
        -------
        A tensor of shape (batch_size, timesteps, output_projection_dim),
        where output_projection_dim = input_dim by default.
        """
        num_heads = self._num_heads

        batch_size, passage_length, _ = passage_tensor.size()
        batch_size, question_length, _ = question_tensor.size()

        if self._use_positional_encoding:
            passage_tensor = add_positional_features(passage_tensor)

        if passage_mask is None:
            passage_mask = passage_tensor.new_ones(batch_size, passage_length)
        if question_mask is None:
            question_mask = question_tensor.new_ones(batch_size, question_length)

        # Shape (batch_size, passage, 2 * input_dim)
        combined_projection = self._norm_layer(passage_tensor)
        combined_projection = self.dropout(combined_projection)
        combined_projection = self._combined_projection(combined_projection)
        combined_projection = self.dropout(combined_projection)

        # Shape (batch_size, passage, input_dim)
        keys, values = combined_projection.split(self._input_dim, -1)
        keys = keys.contiguous()
        queries = question_tensor.contiguous()
        values = values.contiguous()
        # Shape (num_heads * batch_size, passage, input_dim / num_heads)
        values_per_head = values.view(batch_size, passage_length, num_heads, int(self._input_dim/num_heads))
        values_per_head = values_per_head.transpose(1, 2).contiguous()
        values_per_head = values_per_head.view(batch_size * num_heads, passage_length, int(self._input_dim/num_heads))

        # Shape (num_heads * batch_size, question, input_dim / num_heads)
        queries_per_head = queries.view(batch_size, question_length, num_heads, int(self._input_dim/num_heads))
        queries_per_head = queries_per_head.transpose(1, 2).contiguous()
        queries_per_head = queries_per_head.view(batch_size * num_heads, question_length, int(self._input_dim/num_heads))

        # Shape (num_heads * batch_size, passage, input_dim / num_heads)
        keys_per_head = keys.view(batch_size, passage_length, num_heads, int(self._input_dim/num_heads))
        keys_per_head = keys_per_head.transpose(1, 2).contiguous()
        keys_per_head = keys_per_head.view(batch_size * num_heads, passage_length, int(self._input_dim/num_heads))

        # shape (num_heads * batch_size, passage, question)
        scaled_similarities = torch.bmm(keys_per_head/ self._scale, queries_per_head.transpose(1, 2))

        # shape (num_heads * batch_size, passage, question)
        # Normalise the distributions, using the same mask for all heads.
        passage_question_attention = masked_softmax(scaled_similarities,
                                                    question_mask.repeat(1, num_heads).view(batch_size * num_heads, question_length),
                                                    memory_efficient=True)
        # Shape: (num_heads * batch_size, passage_length, queries_dim/num_heads)
        passage_question_vectors = weighted_sum(queries_per_head, passage_question_attention)

        # shape (num_heads * batch_size, question, passage)
        # Normalise the distributions, using the same mask for all heads.
        question_passage_attention = masked_softmax(scaled_similarities.transpose(1, 2),
                                                    passage_mask.repeat(1, num_heads).view(batch_size * num_heads, passage_length),
                                                    memory_efficient=True)

        # Shape: (num_heads * batch_size, passage_length, passage_length)
        attention_over_attention = torch.bmm(passage_question_attention, question_passage_attention)
        attention_over_attention = self.dropout(attention_over_attention)
        # Shape: (num_heads * batch_size, passage_length, input_dim/num_heads)
        passage_passage_vectors = weighted_sum(values_per_head, attention_over_attention)


        # Reshape back to original shape (batch_size, passage_length, input_dim)
        # shape (batch_size, num_heads, passage_length, input_dim/num_heads)
        passage_passage_vectors = passage_passage_vectors.view(batch_size, num_heads, passage_length, int(self._input_dim / num_heads))
        # shape (batch_size, passage_length, num_heads, input_dim/num_heads)
        passage_passage_vectors = passage_passage_vectors.transpose(1, 2).contiguous()
        # shape (batch_size, passage_length, input_dim)
        passage_passage_vectors = passage_passage_vectors.view(batch_size, passage_length, self._input_dim)

        # Reshape back to original shape (batch_size, passage_length, input_dim)
        # shape (batch_size, num_heads, passage_length, input_dim/num_heads)
        passage_question_vectors = passage_question_vectors.view(batch_size, num_heads, passage_length, int(self._input_dim / num_heads))
        # shape (batch_size, passage_length, num_heads, input_dim/num_heads)
        passage_question_vectors = passage_question_vectors.transpose(1, 2).contiguous()
        # shape (batch_size, passage_length, input_dim)
        passage_question_vectors = passage_question_vectors.view(batch_size, passage_length, self._input_dim)

        #[QAd;P*QAd;P*AqAd]
        merged_passage_attention_vectors = self._merged_norm_layer(torch.cat([passage_tensor,
                        passage_question_vectors,
                       passage_tensor * passage_question_vectors,
                       passage_tensor * passage_passage_vectors],
                      dim=-1))


        # shape (batch_size, timesteps, input_size*3)
        merged_passage_attention_vectors = self.dropout(merged_passage_attention_vectors)
        output = self._output_projection(merged_passage_attention_vectors)
        output = output + passage_tensor
        output = self._out_norm_layer(output)

        return output
