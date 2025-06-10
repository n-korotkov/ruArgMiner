import re
import numpy as np
from numpy.random import PCG64
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

from corpus_utils import *
from segmentation import *
from simple_structure import *

def count_cyrillic(s):
    return sum([c in 'абвгдежзийклмнопрстуфхцчшщъыьэюя' for c in s])

def filter_pred(s):
    return s.count(' ') > 1 and count_cyrillic(s) / len(s) > 2/3

def preprocess_text(s):
    s = re.sub(r'[\W\d]+', ' ', s)
    s = s.lower()
    s = s.replace("ё", "е")
    s = s.strip()
    return s

def add_contexts(st: SimpleArgStructure, i, ctx_len):
    s, e = st.paragraphs[st.fragments[i].paragraph_index].fragment_span
    context_left = '.'.join([st.fragments[j].content() for j in range(max(i - ctx_len, s), i)])
    context_right = '.'.join([st.fragments[j].content() for j in range(i + 1, min(i + ctx_len + 1, e))])
    return [
        preprocess_text(context_left),
        preprocess_text(st.fragments[i].content()),
        preprocess_text(context_right)
    ]

def triplets_from_simple_structures(simple_structures: list[SimpleArgStructure], ctx_len, random_state):
    np_rng = np.random.Generator(PCG64(random_state))

    data = []
    for st in simple_structures:
        filter_fragment = lambda i: filter_pred(preprocess_text(st.fragments[i].content()))
        st_data = []

        for para in st.paragraphs:
            s, e = para.fragment_span
            candidate_fragments = list(filter(filter_fragment, range(s, e)))
            for _ in range(int(len(candidate_fragments) ** 0.5)):
                np_rng.shuffle(candidate_fragments)
                for anchor, pos in zip(candidate_fragments[::2], candidate_fragments[1::2]):
                    neg = pos
                    while neg in st.related_fragments[anchor] or s <= neg < e or not filter_fragment(neg):
                        neg = np_rng.integers(0, len(st.fragments))
                    st_data.append([anchor, pos, neg])

        for anchor in filter(filter_fragment, range(len(st.fragments))):
            for pos in filter(filter_fragment, st.related_fragments[anchor]):
                neg = pos
                while neg in st.related_fragments[anchor] or not filter_fragment(neg):
                    neg = np_rng.integers(0, len(st.fragments))
                st_data.append([anchor, pos, neg])

        for anchor, pos, neg in st_data:
            data.append(add_contexts(st, anchor, ctx_len) + add_contexts(st, pos, ctx_len) + add_contexts(st, neg, ctx_len))

    column_names = ['anchor_left', 'anchor', 'anchor_right', 'positive_left', 'positive', 'positive_right', 'negative_left', 'negative', 'negative_right']
    return Dataset.from_dict(dict(zip(column_names, zip(*data))))

def transitive_triplets_from_simple_structures(simple_structures: list[SimpleArgStructure], ctx_len, random_state):
    np_rng = np.random.Generator(PCG64(random_state))

    data = []
    for st in simple_structures:
        filter_fragment = lambda i: filter_pred(preprocess_text(st.fragments[i].content()))

        for anchor in filter(filter_fragment, range(len(st.fragments))):
            for pos in filter(filter_fragment, st.related_fragments[anchor]):
                indirectly_related = list(filter(filter_fragment, st.related_fragments[pos] - st.related_fragments[anchor] - set([anchor])))
                if len(indirectly_related) > 0:
                    neg = np_rng.choice(indirectly_related)
                    data.append(add_contexts(st, anchor, ctx_len) + add_contexts(st, pos, ctx_len) + add_contexts(st, neg, ctx_len))

    column_names = ['anchor_left', 'anchor', 'anchor_right', 'positive_left', 'positive', 'positive_right', 'negative_left', 'negative', 'negative_right']
    return Dataset.from_dict(dict(zip(column_names, zip(*data))))

def link_pairs_from_simple_structures(simple_structures: list[SimpleArgStructure], ctx_len, random_state):
    np_rng = np.random.Generator(PCG64(random_state))

    data = []
    for st in simple_structures:
        filter_fragment = lambda i: filter_pred(preprocess_text(st.fragments[i].content()))
        st_data = []

        for i in st.supports:
            for j in st.supports[i]:
                st_data.append([i, j, 'support'])

        for i in st.attacks:
            for j in st.attacks[i]:
                st_data.append([i, j, 'attack'])

        for _ in range(len(st_data)):
            i, j = np_rng.integers(0, len(st.fragments), 2)
            while (i in st.supports and j in st.supports[i]) or (i in st.attacks and j in st.attacks[i]):
                i, j = np_rng.integers(0, len(st.fragments), 2)
            st_data.append([i, j, 'none'])

        for i, j, label in st_data:
            if filter_fragment(i) and filter_fragment(j):
                data.append(add_contexts(st, i, ctx_len) + add_contexts(st, j, ctx_len) + [label])

    column_names=['fragment1_left', 'fragment1', 'fragment1_right', 'fragment2_left', 'fragment2', 'fragment2_right', 'label']
    return Dataset.from_dict(dict(zip(column_names, zip(*data))))

def random_pairs_from_simple_structures(simple_structures: list[SimpleArgStructure], ctx_len, random_state):
    np_rng = np.random.Generator(PCG64(random_state))

    data = []
    for st in simple_structures:
        filter_fragment = lambda i: filter_pred(preprocess_text(st.fragments[i].content()))

        for _ in range(1000):
            i, j = np_rng.choice(range(len(st.fragments)), 2, replace=False)
            if filter_fragment(i) and filter_fragment(j):
                data.append(add_contexts(st, i, ctx_len) + add_contexts(st, j, ctx_len) + [i in st.related_fragments[j]])

    column_names = ['fragment1_left', 'fragment1', 'fragment1_right', 'fragment2_left', 'fragment2', 'fragment2_right', 'related']
    return Dataset.from_dict(dict(zip(column_names, zip(*data))))

def build_dataset(simple_structures, row_selection_fn, eval_size, test_size, ctx_len=0, random_state=42):
    train_structures, test_structures = train_test_split(simple_structures, test_size=test_size, random_state=random_state)
    train_structures, eval_structures = train_test_split(train_structures, test_size=eval_size / (1 - test_size), random_state=random_state)

    return DatasetDict({
        'train': row_selection_fn(train_structures, ctx_len, random_state),
        'eval': row_selection_fn(eval_structures, ctx_len, random_state),
        'test': row_selection_fn(test_structures, ctx_len, random_state),
    })

def merge_datasets(dataset0, dataset1):
    def merge_split(split0, split1):
        return Dataset.from_dict({column: split0[column] + split1[column] for column in split0.column_names})

    return DatasetDict({
        'train': merge_split(dataset0['train'], dataset1['train']),
        'eval': merge_split(dataset0['eval'], dataset1['eval']),
        'test': merge_split(dataset0['test'], dataset1['test']),
    })
