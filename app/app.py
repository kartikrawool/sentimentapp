from flask import Flask
from joblib import load 
import numpy as np
import pandas as pd 

app = Flask(__name__)


model_in = load('svcmodel.joblib')

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"