import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('login.html')


@app.route('/predict', methods=['POST'])
def predict():
	return render_template('index.html')

if __name__ == "__main__":
    app.run(debug = True)
