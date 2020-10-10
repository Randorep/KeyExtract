from flask_restful import reqparse, abort, Api, Resource
from flask import Flask, render_template, url_for, make_response,jsonify,request
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sentencepiece as spm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



app = Flask(__name__,template_folder='templates')


module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")
input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
encodings = module(
    inputs=dict(
        values=input_placeholder.values,
        indices=input_placeholder.indices,
        dense_shape=input_placeholder.dense_shape))

with tf.Session() as sess:
  spm_path = sess.run(module(signature="spm_path"))

sp = spm.SentencePieceProcessor()
sp.Load(spm_path)
print("SentencePiece model loaded at {}.".format(spm_path))

def process_to_IDs_in_sparse_format(sp, sentences):
  # An utility method that processes sentences with the sentence piece processor
  # 'sp' and returns the results in tf.SparseTensor-similar format:
  # (values, indices, dense_shape)
  ids = [sp.EncodeAsIds(x) for x in sentences]
  max_len = max(len(x) for x in ids)
  dense_shape=(len(ids), max_len)
  values=[item for sublist in ids for item in sublist]
  indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
  return (values, indices, dense_shape)


word = "food"
word2 = "burger"
messages = [word, word2]


values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, messages)



with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    message_embeddings = session.run(
        encodings,
        feed_dict={input_placeholder.values: values,
                    input_placeholder.indices: indices,
                    input_placeholder.dense_shape: dense_shape})


def process_to_IDs_in_sparse_format(sp, sentences):
  # An utility method that processes sentences with the sentence piece processor
  # 'sp' and returns the results in tf.SparseTensor-similar format:
  # (values, indices, dense_shape)
  ids = [sp.EncodeAsIds(x) for x in sentences]
  max_len = max(len(x) for x in ids)
  dense_shape=(len(ids), max_len)
  values=[item for sublist in ids for item in sublist]
  indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
  return (values, indices, dense_shape)

similarity_matrix = cosine_similarity(message_embeddings)

@app.route('/')
def main():
    a = similarity_matrix
    b = str(a[1])
    c = b.replace("[","")
    c = c.replace("]","")
    c = c.split()
    return str((float(c[0])+float(c[1]))/2*100)




app.run(debug=True)