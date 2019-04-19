# pylint: disable=no-self-use,invalid-name
import numpy
import pytest
import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from mymodel.coattention import CoattentionEncoder
from allennlp.modules.similarity_functions import MultiHeadedSimilarity

class TestCoattentionEncoder(AllenNlpTestCase):
    def test_get_dimension_is_correct(self):
        encoder = CoattentionEncoder(input_dim=5, combination='1,2')
        assert encoder.get_input_dim() == 5
        assert encoder.get_output_dim() == 10
        # encoder = CoattentionEncoder(input_dim=5, combination='1+2')
        # assert encoder.get_input_dim() == 5
        # assert encoder.get_output_dim() == 5

        params = Params({'input_dim': 8,
                         'num_attention_heads': 4,
                         'similarity_function': {'type': 'bilinear',
                                                 'tensor_1_dim': 8},
                         'combination': '1,2'})
        encoder = CoattentionEncoder.from_params(params)
        assert encoder.get_input_dim() == 8
        assert encoder.get_output_dim() == 16

    def test_constructor_asserts_multi_head_consistency(self):

        similarity = MultiHeadedSimilarity(3, 6)
        with pytest.raises(ConfigurationError) as exception_info:
            CoattentionEncoder(input_dim=5, similarity_function=similarity)
        assert 'Cannot use MultiHead' in exception_info.value.message

    def test_forward_works_with_simple_attention(self):
        # We're not going to check the output values here, as that's complicated; we'll just make
        # sure the code runs and the shapes are correct.
        encoder = CoattentionEncoder(input_dim=5)
        dst_tensor = torch.from_numpy(numpy.random.rand(4, 3, 5)).float()
        att_tensor = torch.from_numpy(numpy.random.rand(4, 2, 5)).float()
        encoder_output = encoder(dst_tensor, att_tensor, None, None)
        assert list(encoder_output.size()) == [4, 3, 15]  # default combination is 1,2

        encoder2 = CoattentionEncoder(input_dim=5, combination='1+2')
        encoder_output = encoder2(dst_tensor, att_tensor, None, None)
        assert list(encoder_output.size()) == [4, 3, 5]  # default combination is 1+2


test = TestCoattentionEncoder()
test.test_forward_works_with_simple_attention()