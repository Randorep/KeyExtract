from flask import Flask, render_template, url_for, make_response,jsonify,request
import tensorflow_hub as hub
import tensorflow as tf

app = Flask(__name__,template_folder='templates')

module_url = ''

@app.route('/')
def main():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)