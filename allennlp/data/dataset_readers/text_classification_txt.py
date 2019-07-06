from typing import Dict
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("text_classification_txt")
class TextClassificationTxtReader(DatasetReader):
    """
    Reads a file from the cnews dataset. This data is formatted as text, one instance per line.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        tokens: ``TextField`` and
        label: ``LabelField``

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for texts.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this for texts.  See :class:`TokenIndexer`.
    max_sequence_length: ``int``, optional (default = ``None``)
        If specified, will truncate tokens to specified maximum length.
    skip_label_indexing: ``bool``, optional (default = ``False``)
        Whether or not to skip label indexing. You might want to skip label indexing if your
        labels are numbers, so the dataset reader doesn't re-number them starting from 0.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be read lazily.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sequence_length: int = None,
                 skip_label_indexing: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._max_sequence_length = max_sequence_length
        self._skip_label_indexing = skip_label_indexing

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as cnews_file:
            logger.info("Reading cnews instances from text dataset at: %s", file_path)
            for line in cnews_file:
                example = line.split(" ")
                if len(example) != 2:
                    continue
                label = example[0]
                text = example[1]

                if self._skip_label_indexing:
                    try:
                        label = int(label)
                    except ValueError:
                        raise ValueError('Labels must be integers if skip_label_indexing is True.')
                else:
                    label = str(label)

                yield self.text_to_instance(text, label)

    def _truncate(self, tokens):
        """
        truncate a set of tokens using the provided sequence length
        """
        if len(tokens) > self._max_sequence_length:
            tokens = tokens[:self._max_sequence_length]
        return tokens

    @overrides
    def text_to_instance(self, text: str, label: Union[str, int] = None) -> Instance:
        """
        Parameters
        ----------
        text : ``str``, required.
            The text to classify
        label : ``str``, optional, (default = None).
            The label for this text.

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence or phrase.
            label : ``LabelField``
                The label label of the sentence or phrase.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        text_tokens = self._tokenizer.tokenize(text)
        if self._max_sequence_length is not None:
                text_tokens = self._truncate(text_tokens)
        fields['text'] = TextField(text_tokens, self._token_indexers)
        if label:
            fields['label'] = LabelField(label,
                                         skip_indexing=self._skip_label_indexing)

        metadata = {"text": [x.text for x in text_tokens]}
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)
