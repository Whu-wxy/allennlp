# pylint: disable=no-self-use,invalid-name
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.seq2seq_encoders import StackedCoattentionEncoder


class TestStackedCoAttention(AllenNlpTestCase):
    def test_get_dimension_is_correct(self):
        encoder = StackedCoattentionEncoder(input_dim=9,
                                              feedforward_hidden_dim=5,
                                              num_layers=3,
                                              num_attention_heads=3)
        assert encoder.get_input_dim() == 9
        # hidden_dim + projection_dim
        assert encoder.get_output_dim() == 9

    def test_stacked_self_attention_can_run_foward(self):
        # Correctness checks are elsewhere - this is just stacking
        # blocks which are already well tested, so we just check shapes.
        encoder = StackedCoattentionEncoder(input_dim=9,
                                            feedforward_hidden_dim=5,
                                              num_layers=3,
                                              num_attention_heads=3)
        passage = torch.randn([3, 10, 9])
        question = torch.randn([3, 5, 9])
        output = encoder(passage, question)
        assert list(output.size()) == [3, 10, 9]
