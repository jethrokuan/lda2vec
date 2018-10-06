"""Evaluates LDA2vec models."""

import tensorflow as tf

with tf.variable_scope("embeddings", reuse=True):
            topic_embedding = tf.get_variable("topic_embedding")
            topic_embedding = tf.nn.l2_normalize(topic_embedding, 1)
            word_embedding = tf.get_variable("word_embedding")
            word_embedding = tf.nn.l2_normalize(word_embedding, 1)

with tf.variable_scope("k_closest"):
            indices = tf.placeholder(tf.int32, shape=[None], name='k_idxs')
            topic = tf.nn.embedding_lookup(topic_embedding, indices)
            cosine_sim = tf.matmul(topic, tf.transpose(word_embedding, [1,0]))
            tf.Print(cosine_sim, [cosine_sim], "cosine_sim")

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "~/models/lda2vec/model.cpkt")
    print(topic_embedding.eval())
    print(word_embedding.eval())
