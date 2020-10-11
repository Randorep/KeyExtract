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

sess = tf.Session()
spm_path = sess.run(module(signature="spm_path"))

sp = spm.SentencePieceProcessor()
sp.Load(spm_path)

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

def process_to_IDs_in_sparse_format(sp, sentences):
  ids = [sp.EncodeAsIds(x) for x in sentences]
  max_len = max(len(x) for x in ids)
  dense_shape=(len(ids), max_len)
  values=[item for sublist in ids for item in sublist]
  indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
  return (values, indices, dense_shape)

def wordchoice(word,word2):
  messages = [word,word2]
  values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, messages)
  message_embeddings = sess.run(
      encodings,
      feed_dict={input_placeholder.values: values,
                  input_placeholder.indices: indices,
                  input_placeholder.dense_shape: dense_shape})
  return message_embeddings




@app.route('/')
def main():
    return render_template("index.html")

@app.route('/calc', methods=['GET','POST'])
def calc():
  arr = []
  if request.method =='POST':
    typeform = str(request.form["tyw"])
    typeform2 = request.form["t2yw"]
    percent = request.form['pyw']
    x = typeform.split()
    for i in x:
      similarity_matrix = cosine_similarity(wordchoice(i,typeform2))
      a = similarity_matrix 
      b = str(a[1])
      c = b.replace("[","")
      c = c.replace("]","")
      c = c.split()
      g = float(c[0])*100
      if g > float(percent):
        arr.append(i.replace(","," "))
      if arr.len() == 0:
        return "None were found"
    return render_template("results.html", arr=arr)



app.run(debug=True)