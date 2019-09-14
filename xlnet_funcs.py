import collections
import re
import numpy as np
import itertools
import tensorflow as tf
from prepro_utils import preprocess_text, encode_ids, encode_pieces
from xlnet_lib import *

SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4

special_symbols = {
    '<unk>': 0,
    '<s>': 1,
    '</s>': 2,
    '<cls>': 3,
    '<sep>': 4,
    '<pad>': 5,
    '<mask>': 6,
    '<eod>': 7,
    '<eop>': 8,
}

UNK_ID = special_symbols['<unk>']
CLS_ID = special_symbols['<cls>']
SEP_ID = special_symbols['<sep>']
MASK_ID = special_symbols['<mask>']
EOD_ID = special_symbols['<eod>']


def generate_ngram(seq, ngram=(1, 3)):
    g = []
    for i in range(ngram[0], ngram[-1] + 1):
        g.extend(list(ngrams_generator(seq, i)))
    return g


def _pad_sequence(
        sequence,
        n,
        pad_left=False,
        pad_right=False,
        left_pad_symbol=None,
        right_pad_symbol=None,
):
    sequence = iter(sequence)
    if pad_left:
        sequence = itertools.chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = itertools.chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams_generator(
        sequence,
        n,
        pad_left=False,
        pad_right=False,
        left_pad_symbol=None,
        right_pad_symbol=None,
):
    """
    generate ngrams.

    Parameters
    ----------
    sequence : list of str
        list of tokenize words.
    n : int
        ngram size

    Returns
    -------
    ngram: list
    """
    sequence = _pad_sequence(
        sequence, n, pad_left, pad_right, left_pad_symbol, right_pad_symbol
    )

    history = []
    while n > 1:
        try:
            next_item = next(sequence)
        except StopIteration:
            return
        history.append(next_item)
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def tokenize_fn(text, sp_model):
    text = preprocess_text(text, lower=False)
    return encode_ids(sp_model, text)


def merge_sentencepiece_tokens(paired_tokens, weighted=True):
    new_paired_tokens = []
    n_tokens = len(paired_tokens)
    rejected = ['<cls>', '<sep>']

    i = 0

    while i < n_tokens:

        current_token, current_weight = paired_tokens[i]
        if not current_token.startswith('▁') and current_token not in rejected:
            previous_token, previous_weight = new_paired_tokens.pop()
            merged_token = previous_token
            merged_weight = [previous_weight]
            while (
                    not current_token.startswith('▁')
                    and current_token not in rejected
            ):
                merged_token = merged_token + current_token.replace('▁', '')
                merged_weight.append(current_weight)
                i = i + 1
                current_token, current_weight = paired_tokens[i]
            merged_weight = np.mean(merged_weight)
            new_paired_tokens.append((merged_token, merged_weight))

        else:
            new_paired_tokens.append((current_token, current_weight))
            i = i + 1

    words = [
        i[0].replace('▁', '')
        for i in new_paired_tokens
        if i[0] not in ['<cls>', '<sep>', '<pad>']
    ]
    weights = [
        i[1]
        for i in new_paired_tokens
        if i[0] not in ['<cls>', '<sep>', '<pad>']
    ]
    if weighted:
        weights = np.array(weights)
        weights = weights / np.sum(weights)
    return list(zip(words, weights))


def xlnet_tokenization(tokenizer, texts):
    input_ids, input_masks, segment_ids, s_tokens = [], [], [], []
    for text in texts:
        tokens_a = tokenize_fn(text, tokenizer)
        tokens = []
        segment_id = []
        for token in tokens_a:
            tokens.append(token)
            segment_id.append(SEG_ID_A)

        tokens.append(SEP_ID)
        segment_id.append(SEG_ID_A)
        tokens.append(CLS_ID)
        segment_id.append(SEG_ID_CLS)

        input_id = tokens
        input_mask = [0] * len(input_id)

        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        s_tokens.append([tokenizer.IdToPiece(i) for i in tokens])

    maxlen = max([len(i) for i in input_ids])
    input_ids = padding_sequence(input_ids, maxlen, padding='pre')
    input_masks = padding_sequence(
        input_masks, maxlen, padding='pre', pad_int=1
    )
    segment_ids = padding_sequence(
        segment_ids, maxlen, padding='pre', pad_int=SEG_ID_PAD
    )

    return input_ids, input_masks, segment_ids, s_tokens


def padding_sequence(seq, maxlen, padding='post', pad_int=0):
    padded_seqs = []
    for s in seq:
        if padding == 'post':
            padded_seqs.append(s + [pad_int] * (maxlen - len(s)))
        if padding == 'pre':
            padded_seqs.append([pad_int] * (maxlen - len(s)) + s)
    return padded_seqs


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match('^(.*):\\d+$', name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name_to_variable[name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ':0'] = 1

    return (assignment_map, initialized_variable_names)


class _Model:
    def __init__(self, xlnet_config, tokenizer, checkpoint, pool_mode='last'):

        kwargs = dict(
            is_training=True,
            use_tpu=False,
            use_bfloat16=False,
            dropout=0.0,
            dropatt=0.0,
            init='normal',
            init_range=0.1,
            init_std=0.05,
            clamp_len=-1,
        )

        xlnet_parameters = RunConfig(**kwargs)

        self._tokenizer = tokenizer
        _graph = tf.Graph()
        with _graph.as_default():
            self.X = tf.placeholder(tf.int32, [None, None])
            self.segment_ids = tf.placeholder(tf.int32, [None, None])
            self.input_masks = tf.placeholder(tf.float32, [None, None])

            xlnet_model = XLNetModel(
                xlnet_config=xlnet_config,
                run_config=xlnet_parameters,
                input_ids=tf.transpose(self.X, [1, 0]),
                seg_ids=tf.transpose(self.segment_ids, [1, 0]),
                input_mask=tf.transpose(self.input_masks, [1, 0]),
            )

            self.logits = xlnet_model.get_pooled_out(pool_mode, True)
            self._sess = tf.InteractiveSession()
            self._sess.run(tf.global_variables_initializer())
            tvars = tf.trainable_variables()
            assignment_map, _ = get_assignment_map_from_checkpoint(
                tvars, checkpoint
            )
            self._saver = tf.train.Saver(var_list=assignment_map)
            attentions = [
                n.name
                for n in tf.get_default_graph().as_graph_def().node
                if 'rel_attn/Softmax' in n.name
            ]
            g = tf.get_default_graph()
            self.attention_nodes = [
                g.get_tensor_by_name('%s:0' % (a)) for a in attentions
            ]

    def vectorize(self, strings):
        """
        Vectorize string inputs using bert attention.

        Parameters
        ----------
        strings : str / list of str

        Returns
        -------
        array: vectorized strings
        """

        if isinstance(strings, list):
            if not isinstance(strings[0], str):
                raise ValueError('input must be a list of strings or a string')
        else:
            if not isinstance(strings, str):
                raise ValueError('input must be a list of strings or a string')
        if isinstance(strings, str):
            strings = [strings]

        input_ids, input_masks, segment_ids, _ = xlnet_tokenization(
            self._tokenizer, strings
        )
        return self._sess.run(
            self.logits,
            feed_dict={
                self.X: input_ids,
                self.segment_ids: segment_ids,
                self.input_masks: input_masks,
            },
        )

    def attention(self, strings, method='last', **kwargs):
        """
        Get attention string inputs from xlnet attention.

        Parameters
        ----------
        strings : str / list of str
        method : str, optional (default='last')
            Attention layer supported. Allowed values:

            * ``'last'`` - attention from last layer.
            * ``'first'`` - attention from first layer.
            * ``'mean'`` - average attentions from all layers.

        Returns
        -------
        array: attention
        """

        if isinstance(strings, list):
            if not isinstance(strings[0], str):
                raise ValueError('input must be a list of strings or a string')
        else:
            if not isinstance(strings, str):
                raise ValueError('input must be a list of strings or a string')
        if isinstance(strings, str):
            strings = [strings]

        method = method.lower()
        if method not in ['last', 'first', 'mean']:
            raise Exception(
                "method not supported, only support ['last', 'first', 'mean']"
            )

        input_ids, input_masks, segment_ids, s_tokens = xlnet_tokenization(
            self._tokenizer, strings
        )
        maxlen = max([len(s) for s in s_tokens])
        s_tokens = padding_sequence(s_tokens, maxlen, pad_int='<cls>')
        attentions = self._sess.run(
            self.attention_nodes,
            feed_dict={
                self.X: input_ids,
                self.segment_ids: segment_ids,
                self.input_masks: input_masks,
            },
        )

        if method == 'first':
            cls_attn = np.transpose(attentions[0][:, 0], (1, 0, 2))

        if method == 'last':
            cls_attn = np.transpose(attentions[-1][:, 0], (1, 0, 2))

        if method == 'mean':
            cls_attn = np.transpose(
                np.mean(attentions, axis=0).mean(axis=1), (1, 0, 2)
            )

        cls_attn = np.mean(cls_attn, axis=1)
        total_weights = np.sum(cls_attn, axis=-1, keepdims=True)
        attn = cls_attn / total_weights
        output = []
        for i in range(attn.shape[0]):
            output.append(
                merge_sentencepiece_tokens(list(zip(s_tokens[i], attn[i])))
            )
        return output


def print_topics_modelling(
    topics, feature_names, sorting, n_words = 20, return_df = True
):
    if return_df:
        try:
            import pandas as pd
        except:
            raise Exception(
                'pandas not installed. Please install it and try again or set `return_df = False`'
            )
    df = {}
    for i in range(topics):
        words = []
        for k in range(n_words):
            words.append(feature_names[sorting[i, k]])
        df['topic %d' % (i)] = words
    if return_df:
        return pd.DataFrame.from_dict(df)
    else:
        return df