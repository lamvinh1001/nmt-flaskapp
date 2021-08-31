import re
import spacy
import pathlib
import dill
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


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


def get_fields():
    with (open('vocab.pkl', 'rb')) as pickel_file:
        # TRG = dill.load(pickel_file)
        SRC, TRG = dill.load(pickel_file)
    return SRC, TRG
