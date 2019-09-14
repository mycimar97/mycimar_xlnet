import re
import collections
import numpy as np
import itertools
from prepro_utils import preprocess_text, encode_ids, encode_pieces
import sentencepiece as spm
from xlnet_lib import *
import xlnet_funcs
import tensorflow as tf
from sklearn.cluster import KMeans
from tqdm import tqdm
from nltk.corpus import stopwords
from pprint import pprint
with open('reviews_as_raw_text.txt') as fopen:
    negative = fopen.read().split('\n')[:-1]
test = []
for x in negative:
    if len(x) < 1000:
        test.append(x)

negative = test

sp_model = spm.SentencePieceProcessor()
sp_model.Load('xlnet_cased_L-12_H-768_A-12/spiece.model')
xlnet_config = XLNetConfig(
    json_path = 'xlnet_cased_L-12_H-768_A-12/xlnet_config.json'
)
xlnet_checkpoint = 'xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt'
model = xlnet_funcs._Model(
    xlnet_config, sp_model, xlnet_checkpoint, pool_mode = 'last'
)
model._saver.restore(model._sess, xlnet_checkpoint)
batch_size = 1
ngram = (1, 3)
n_topics = 3
rows, attentions = [], []
for i in tqdm(range(0, len(negative), batch_size)):
    index = min(i + batch_size, len(negative))
    rows.append(model.vectorize(negative[i:index]))
    attentions.extend(model.attention(negative[i:index]))
stopwords = set(stopwords.words("english"))
concat = np.concatenate(rows, axis = 0)
kmeans = KMeans(n_clusters = n_topics, random_state = 0).fit(concat)
labels = kmeans.labels_

overall, filtered_a = [], []
for a in attentions:
    f = [i for i in a if i[0] not in stopwords]
    overall.extend(f)
    filtered_a.append(f)

o_ngram = xlnet_funcs.generate_ngram(overall, ngram)
features = []
for i in o_ngram:
    features.append(' '.join([w[0] for w in i]))
features = list(set(features))

components = np.zeros((n_topics, len(features)))
for no, i in enumerate(labels):
    if (no + 1) % 500 == 0:
        print('processed %d'%(no + 1))
    f = xlnet_funcs.generate_ngram(filtered_a[no], ngram)
    for w in f:
        word = ' '.join([r[0] for r in w])
        score = np.mean([r[1] for r in w])
        if word in features:
            components[i, features.index(word)] += score


pprint(xlnet_funcs.print_topics_modelling(
    n_topics,
    feature_names = np.array(features),
    sorting = np.argsort(components)[:, ::-1],
    n_words = 10,
    return_df = True,
))