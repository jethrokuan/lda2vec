"""Text Preprocessor class."""

from collections import Counter
import numpy as np
import re
from tqdm import tqdm
import spacy

def preprocess(docs,
               min_word_count,
               max_word_count,
               min_word_length,
               max_doc_length,
               min_doc_length):
    """Tokenize, clean andd encode documents.

    Args:
        docs (tuple): A list of tuples (document_id, document_text).
        min_word_count (int): the minimum count for a word.
        max_word_count (int): the maximum count for a word.
        min_doc_length (int): the minimum document length.

    Returns:
        encoded_docs (tuple): A list of tuples (document_id, [int]), with the
           list of int representing words in the document.
        id2word (dict): Dictionary mapping index to word.
        word_counts ([int]): word_counts[i] gives the number of occurences
           of word id2word[i].
    """
    nlp = spacy.load("en")

    def clean_and_tokenize(doc):
        text = ' '.join(doc.split())  # remove excessive spaces
        text = nlp(text)
        return [
            t.lemma_ for t in text
            if t.is_alpha and len(t) > 2 and not t.is_stop
        ]

    def _count_unique_tokens(tokenized_docs):
        tokens = []
        for i, doc in tokenized_docs:
            tokens += doc
        return Counter(tokens)

    def _remove_tokens(tokenized_docs, min_count, max_count):
        tokens = [token for _, doc in tokenized_docs for token in doc]
        counter = Counter(tokens)
        keep = {}
        for token, count in counter.most_common():
            keep[token] = count >= min_count or count <= max_count

        return [(i, [t for t in doc if keep[t]]) for i, doc in tokenized_docs]

    def _create_dictionary(tokenized_docs):
        tokens = [token for _, doc in tokenized_docs for token in doc]
        counter = Counter(tokens)
        id2word = {}
        word2id = {}
        word_counts = []

        i = 0
        for token, count in counter.most_common():
            word2id[token] = i
            id2word[i] = token
            word_counts.append(count)
            i += 1

        return word2id, id2word, word_counts

    def _encode_doc(doc, word2id):
        return [word2id[w] for w in doc]

    tokenized_docs = [(i, clean_and_tokenize(doc)) for i, doc in docs]

    # remove short documents
    tokenized_docs = [(i, doc) for i, doc in tokenized_docs
                      if len(doc) >= min_doc_length]

    # remove frequent and infrequent tokens, and remove documents again
    tokenized_docs = _remove_tokens(tokenized_docs, min_word_count,
                                    max_word_count)
    tokenized_docs = [(i, doc) for i, doc in tokenized_docs
                      if len(doc) >= min_doc_length]

    word2id, id2word, word_counts = _create_dictionary(tokenized_docs)

    encoded_docs = [(i, _encode_doc(doc, word2id))
                    for i, doc in tokenized_docs]
    return encoded_docs, id2word, word_counts
