from overrides import overrides
import torch
from torch.nn import Linear

from allennlp.common.checks import ConfigurationError
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.modules.similarity_functions import DotProductSimilarity, SimilarityFunction
from allennlp.modules.similarity_functions import MultiHeadedSimilarity
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

from allennlp.nn import util

@Seq2SeqEncoder.register("coattention_encoder")
class CoattentionEncoder(Seq2SeqEncoder):
    def __init__(self,
                 input_dim: int,
                 similarity_function: SimilarityFunction = DotProductSimilarity(),
                 combination: str = '1,2',
                 output_dim: int = None) -> None:
        super(CoattentionEncoder, self).__init__()
        self._input_dim = input_dim
        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        if isinstance(similarity_function, MultiHeadedSimilarity):
                raise ConfigurationError("Cannot use MultiHead")

        if combination[1] == '+':
            self._combine_projection = Linear(input_dim*2, input_dim)
        elif combination[1] == ',':
            self._combine_projection = lambda x: x

        self._combination = combination
        combined_dim = util.get_combined_dim(combination, [input_dim, input_dim])

        if output_dim:
            self._output_projection = Linear(combined_dim, output_dim)
            self._output_dim = output_dim
        else:
            self._output_projection = lambda x: x
            self._output_dim = combined_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return False

    @overrides
    def forward(self, dst_tokens: torch.Tensor, att_tokens:torch.Tensor,
                 dst_mask: torch.Tensor, att_mask: torch.Tensor):
        """
        DCN : Dynamic Coattention Networks

        :param dst_tokens: B*D*d
        :param att_tokens: B*Q*d
        :return:
        """

        # B*Q*D
        L = self._matrix_attention(att_tokens, dst_tokens)

        # B*Q*D
        Aq = util.masked_softmax(L.contiguous(), att_mask)
        # B*Q*D
        Ad = util.masked_softmax(L.contiguous(), dst_mask)

        # B*Q*d
        Cq = util.weighted_sum(dst_tokens, Aq) #L.bmm(dst_tokens)

        # B*Q*2d
        Q_Cq = torch.cat((att_tokens, Cq), dim=2)
        # B*D*Q
        Ad = torch.transpose(Ad, 1, 2)
        # B*D*2d
        Cd = util.weighted_sum(Q_Cq, Ad)

        Cd = self._combine_projection(Cd)
        # combine origin D and D with question characteristic.
        # Shape: (batch_size, sequence_length, combination_dim)
        combined_tensors = util.combine_tensors(self._combination, [dst_tokens, Cd])
        return self._output_projection(combined_tensors)