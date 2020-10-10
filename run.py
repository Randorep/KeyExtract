from flask import Flask, render_template, url_for, make_response,jsonify,request
import tensorflow_hub as hub
import tensorflow as tf

app = Flask(__name__,template_folder='templates')


@app.route('/')
def main():
    return render_template('static/css/index.css')