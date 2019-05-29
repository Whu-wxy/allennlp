from typing import Any, Dict, List, Optional

import torch
from torch.nn.functional import nll_loss
from torch.nn import LayerNorm

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1
from allennlp.nn.util import masked_softmax


@Model.register("pure_multico")
class PureMultiCo(Model):
    """
    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
    num_highway_layers : ``int``
        The number of highway layers to use in between embedding the input and passing it through
        the phrase layer.
    phrase_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and doing the passage-question attention.
    coattention_layer : ``Seq2SeqEncoder``
        The matrix attention function that we will use when comparing encoded passage and question
        representations.
    modeling_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between the bidirectional
        attention and predicting span start and end.
    dropout_prob : ``float``, optional (default=0.1)
        If greater than 0, we will apply dropout with this probability between layers.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 num_highway_layers: int,
                 coattention_layer: Seq2SeqEncoder,
                 dropout_prob: float = 0.1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        text_embed_dim = text_field_embedder.get_output_dim()
        coatt_in_dim = coattention_layer.get_input_dim()
        coatt_out_dim = coattention_layer.get_output_dim()

        self._text_field_embedder = text_field_embedder

        # self._embedding_proj_layer = torch.nn.Linear(text_embed_dim, coatt_in_dim)
        self._highway_layer = Highway(coatt_in_dim, num_highway_layers)

        self._multihead_coattention_layer = coattention_layer

        self._modeling_proj_layer = torch.nn.Linear(coatt_out_dim + phrase_out_dim * 4, modeling_in_dim)

        self._span_predictor = torch.nn.Linear(coatt_out_dim, 2)

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._metrics = SquadEmAndF1()
        self._dropout = torch.nn.Dropout(p=dropout_prob) if dropout_prob > 0 else lambda x: x

        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.
        span_start : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            beginning position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        span_end : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            ending position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question tokens, passage tokens, original passage
            text, and token offsets into the passage for each instance in the batch.  The length
            of this list should be the batch size, and each dictionary should have the keys
            ``question_tokens``, ``passage_tokens``, ``original_passage``, and ``token_offsets``.

        Returns
        -------
        An output dictionary consisting of:
        span_start_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span start position.
        span_start_probs : torch.FloatTensor
            The result of ``softmax(span_start_logits)``.
        span_end_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span end position (inclusive).
        span_end_probs : torch.FloatTensor
            The result of ``softmax(span_end_logits)``.
        best_span : torch.IntTensor
            The result of a constrained inference over ``span_start_logits`` and
            ``span_end_logits`` to find the most probable span.  Shape is ``(batch_size, 2)``
            and each offset is a token index.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        best_span_str : List[str]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        """
        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(passage).float()

        embedded_question = self._dropout(self._text_field_embedder(question))
        embedded_passage = self._dropout(self._text_field_embedder(passage))
        embedded_question = self._highway_layer(self._embedding_proj_layer(embedded_question))
        embedded_passage = self._highway_layer(self._embedding_proj_layer(embedded_passage))

        batch_size = embedded_question.size(0)

        # Shape: (batch_size, passage_length, dim)
        coattention_vectors = self._multihead_coattention_layer(embedded_passage, embedded_question, passage_mask,
                                                                question_mask)
        if not self.training:
            # Shape: (batch_size, passage_length, question_length)
            passage_question_attention = coattention_vectors.bmm(embedded_question.transpose(1, 2))
        elif self.training:
            passage_question_attention = None

        # Shape: (batch_size, passage_length, 2))
        span_logits = self._span_predictor(coattention_vectors).squeeze(-1)

        # Shape: (batch_size, passage_length, 1)
        span_start_logits, span_end_logits = span_logits.split(1, dim=-1)
        # Shape: (batch_size, passage_length)
        span_start_logits = span_start_logits.squeeze(-1)
        span_end_logits = span_end_logits.squeeze(-1)

        # Shape: (batch_size, passage_length, modeling_dim * 2)
        span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e32)
        span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e32)

        # Shape: (batch_size, passage_length)
        span_start_probs = torch.nn.functional.softmax(span_start_logits, dim=-1)
        span_end_probs = torch.nn.functional.softmax(span_end_logits, dim=-1)

        best_span = get_best_span(span_start_logits, span_end_logits)

        output_dict = {
            "passage_question_attention": passage_question_attention,
            "span_start_logits": span_start_logits,
            "span_start_probs": span_start_probs,
            "span_end_logits": span_end_logits,
            "span_end_probs": span_end_probs,
            "best_span": best_span,
        }

        # Compute the loss for training.
        if span_start is not None:
            loss = nll_loss(util.masked_log_softmax(span_start_logits, passage_mask), span_start.squeeze(-1))
            self._span_start_accuracy(span_start_logits, span_start.squeeze(-1))
            loss += nll_loss(util.masked_log_softmax(span_end_logits, passage_mask), span_end.squeeze(-1))
            self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
            self._span_accuracy(best_span, torch.stack([span_start, span_end], -1))
            output_dict["loss"] = loss

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            output_dict['best_span_str'] = []
            question_tokens = []
            passage_tokens = []
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
                passage_str = metadata[i]['original_passage']
                offsets = metadata[i]['token_offsets']
                predicted_span = tuple(best_span[i].detach().cpu().numpy())
                start_offset = offsets[predicted_span[0]][0]
                end_offset = offsets[predicted_span[1]][1]
                best_span_string = passage_str[start_offset:end_offset]
                output_dict['best_span_str'].append(best_span_string)
                answer_texts = metadata[i].get('answer_texts', [])
                if answer_texts:
                    self._metrics(best_span_string, answer_texts)
            output_dict['question_tokens'] = question_tokens
            output_dict['passage_tokens'] = passage_tokens
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._metrics.get_metric(reset)
        return {
            'start_acc': self._span_start_accuracy.get_metric(reset),
            'end_acc': self._span_end_accuracy.get_metric(reset),
            'span_acc': self._span_accuracy.get_metric(reset),
            'em': exact_match,
            'f1': f1_score,
        }
