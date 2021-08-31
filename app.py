import torch
from flask import Flask, render_template, request
from flask_cors import cross_origin
from predict import predict_sentence
from googletts import speak
import re
import spacy
import gc

gc.collect()
torch.cuda.empty_cache()


class tokenize(object):

    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def tokenizer(self, sentence):
        sentence = re.sub(
            r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]


app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/after', methods=['POST', 'GET'])
@cross_origin()
def after():
    en_text = request.form['english']
    vi_text = predict_sentence(en_text)
    return render_template('index1.html', en_input=en_text, vi_output=vi_text)


if __name__ == "__main__":
    app.run(debug=True)
