# pylint: disable=invalid-name,no-self-use,too-many-public-methods
import numpy
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.seq2seq_encoders import MultiHeadCoAttention
from allennlp.common.params import Params


class MultiHeadCoAttentionTest(AllenNlpTestCase):

    def test_multi_head_coattention_can_build_from_params(self):
        params = Params({"num_heads": 2, "input_dim": 4})

        encoder = MultiHeadCoAttention.from_params(params)
        assert isinstance(encoder, MultiHeadCoAttention)
        assert encoder.get_input_dim() == 4
        assert encoder.get_output_dim() == 4

    def test_multi_head_coattention_runs_forward(self):
        attention = MultiHeadCoAttention(num_heads=3,
                                           input_dim=9)
        passage = torch.randn(2, 12, 9)
        question = torch.randn(2, 5, 9)
        assert list(attention(passage, question).size()) == [2, 12, 9]

    def test_multi_head_coattention_respects_masking(self):
        attention = MultiHeadCoAttention(num_heads=3,
                                           input_dim=6,
                                           attention_dropout_prob=0.0)
        passage = torch.randn(2, 12, 6)
        question = torch.randn(2, 5, 6)
        passage_mask = torch.ones([2, 12])
        passage_mask[0, 6:] = 0
        question_mask = torch.ones([2, 5])
        result1, result2 = attention(passage, question, passage_mask, question_mask)
        # Compute the same function without a mask, but with
        # only the unmasked elements - should be the same.
        result_without_mask = attention(passage[:, :6, :], question)
        numpy.testing.assert_almost_equal(result1[0, :6, :].detach().cpu().numpy(),
                                          result_without_mask[0, :, :].detach().cpu().numpy())

test = MultiHeadCoAttentionTest()
test.test_multi_head_coattention_respects_masking()
test.test_multi_head_coattention_runs_forward()