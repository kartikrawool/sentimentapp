from flask import Flask, render_template, request
from joblib import load 
import numpy as np
import pandas as pd 
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
app = Flask(__name__)

app.debug = True




@app.route("/", methods=['GET', 'POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html')
    else:
        text = request.form['text']
        model = load('linearsvcmodel.joblib')
        #model = pickle.load(open('model.pkl','rb'))
        cleaned_text = tweet_cleaner_updated(text)
        sentiment = model.predict([cleaned_text])
        if sentiment[0] == 1:
            senti = "positive"
        else:
            senti = "negative"
        return render_template('index.html', cleaned_tweet = cleaned_text, sentiment = senti)




def tweet_cleaner_updated(text):
    tok = WordPunctTokenizer()

    pat1 = r'@[A-Za-z0-9_]+'
    pat2 = r'https?://[^ ]+'
    pat3 = r'RT'
    pat4 = r'coronavirus'#removing this words as it is related to topic and we dont need it for sentiment
    pat5 = r'corona'
    pat6 = r'virus'
    combined_pat = r'|'.join((pat1, pat2,pat3))
    combined_pat2 = r'|'.join((pat4, pat5,pat6))
    www_pat = r'www.[^ ]+'
    negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                    "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                    "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                    "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                    "mustn't":"must not"}
    neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    stripped = re.sub(combined_pat2, '', lower_case)
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], stripped)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()

