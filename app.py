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
  ids = [sp.EncodeAsIds(x) for x in sentences]
  max_len = max(len(x) for x in ids)
  dense_shape=(len(ids), max_len)
  values=[item for sublist in ids for item in sublist]
  indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
  return (values, indices, dense_shape)

def getmessage(word,word2):
  messages = [word, word2]
  values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, messages)
  with tf.Session() as session:
      session.run([tf.global_variables_initializer(), tf.tables_initializer()])
      message_embeddings = session.run(
          encodings,
          feed_dict={input_placeholder.values: values,
                      input_placeholder.indices: indices,
                      input_placeholder.dense_shape: dense_shape})
  return message_embeddings


def process_to_IDs_in_sparse_format(sp, sentences):
  ids = [sp.EncodeAsIds(x) for x in sentences]
  max_len = max(len(x) for x in ids)
  dense_shape=(len(ids), max_len)
  values=[item for sublist in ids for item in sublist]
  indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
  return (values, indices, dense_shape)



@app.route('/',methods=['GET','POST'])
def main():
  arr = []
  if request.method =='POST':
    typeform = request.form["tyw"]
    typeform2 = request.form["t2yw"]
    percent = request.form['pyw']
    x = typeform.split("")
    for i in x:
      similarity_matrix = cosine_similarity(getmessage(i,typeform2))
      if similarity_matrix > percent:
        arr.append(i)
    return render_template("index.html",arr = arr)
  else:
    return render_template("index.html")




app.run(debug=True)