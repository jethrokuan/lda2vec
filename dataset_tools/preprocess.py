"""Text Preprocessor class."""

from collections import Counter
import numpy as np
import spacy
import tensorflow as tf
from spacy.attrs import LOWER, LIKE_URL, LIKE_EMAIL, IS_OOV, IS_PUNCT


class NlpPipeline(object):
    def __init__(self,
                 documents,
                 max_length=10000,
                 skip_token="<SKIP>",
                 oov_token="<OOV>"):
        self.documents = documents
        self.num_documents = len(documents)
        self.max_length = max_length
        self.nlp = spacy.load("en_core_web_sm")

        # Add the skip_token and oov_token
        self.skip_token = skip_token
        self.oov_token = oov_token
        self.nlp.vocab.strings.add(skip_token)
        self.nlp.vocab.strings.add(oov_token)
        self.skip = self.nlp.vocab.strings[skip_token]
        self.oov = self.nlp.vocab.strings[oov_token]

        # Initialize tokenized_docs
        self.tokenized_docs = np.zeros(
            ((len(self.documents), max_length)), dtype=np.uint64)
        self.tokenized_docs[:] = self.skip

    def _merge_doc(self, doc, bad_deps=["amod", "compound"]):
        for phrase in doc.noun_chunks:
            while len(phrase) > 1 and phrase[0].dep_ not in bad_deps:
                phrase = phrase[1:]
            if len(phrase) > 1:
                phrase.merge(
                    tag=phrase.root.tag_,
                    lemma=phrase.text,
                    ent_type=phrase.root.ent_type_)
        for ent in doc.ents:
            if len(ent) > 1:
                doc.merge(
                    start_idx=ent[0].idx,
                    end_idx=ent[-1].idx,
                    tag=ent.root.tag_,
                    lemma="_".join([token.text for token in ent]),
                    ent_type=ent[0].ent_type_)

        # for token in doc:
        #     print(token)
        #     text = token.text.replace(" ", "_")
        #     if token.is_oov:
        #         print(token.lower_)
        #         self.nlp.vocab.strings.add(token.lower_)

        return doc

    def tokenize(self):
        for row, document in enumerate(self.nlp.pipe(self.documents)):
            document = self._merge_doc(document)
            a = []
            data = document.to_array(
                [LOWER, LIKE_EMAIL, LIKE_URL, IS_OOV, IS_PUNCT])
            if len(data) > 0:
                # Indices to replace with skip token
                # TODO: some issues with is_oov on models
                skip_idx = (data[:, 1] > 0) | (data[:, 2] > 0)

                # oov_idx = (data[:, 3] > 0)

                data[skip_idx] = self.skip
                # data[oov_idx] = self.oov

                # Delete punctuation
                data = np.delete(data, np.where(data[:, 4] == 1), 0)
                length = min(len(data), self.max_length)
                self.tokenized_docs[row, :length] = data[:length, 0].ravel()
        uniques = np.unique(self.tokenized_docs.ravel())
        self.vocab = {v: self.nlp.vocab[v].lower_ for v in uniques}
        self.vocab[self.skip] = self.skip_token
        # self.vocab[self.oov] = self.oov_token

    def compact_documents(self):
        uniques, self.token_counts = np.unique(
            self.tokenized_docs.ravel(), return_counts=True)
        frequencies, hash_ids = zip(*(
            (c, x)
            for c, x in sorted(zip(self.token_counts, uniques), reverse=True)))
        word_ids = np.arange(len(hash_ids))
        self.hash_to_idx = dict(zip(hash_ids, word_ids))
        self.idx_to_hash = dict(zip(word_ids, hash_ids))
        self.idx_to_word = {
            i: self.vocab[self.idx_to_hash[i]]
            for i in word_ids
        }
        self.word_to_idx = {v: k for k, v in self.idx_to_word.items()}
        self.compact_docs = np.vectorize(self.hash_to_idx.get)(
            self.tokenized_docs)
