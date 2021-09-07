import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = ['sentida','danlp','allennlp','hisia','afinn']
# installing packages
for package in packages:
    install(package)

import nltk,numpy as np
nltk.download('punkt')
import operator
import time
class DANISH_SENTIMENT():
  def __init__(self,hisia=True):
    try:
      from afinn import Afinn
      self.afinn = Afinn()
    except:
      print('afinn not installed')
      self.afinn = False
    try:
      from sentida import Sentida
      self.sent = Sentida()
    except:
      print('sentida not loading')
      self.sent = False
    try:
      from danlp.models import load_bert_emotion_model
      self.classifier = load_bert_emotion_model()
    except:
      self.classifier = False
      print('bert emotion not loading')

    try:
      from danlp.models import load_bert_tone_model
      self.classifier_tone = load_bert_tone_model()
    except:
      print('bert tone not working')
      self.classifier_tone = False
    try:
      from danlp.models import load_spacy_model
      self.nlp = load_spacy_model(textcat='sentiment',vectorError=True) # if you got an error saying da.vectors not found, try setting vectorError=True - it is an temp fix
    except:
      print('spacy sentiment not working')
      self.nlp = False
    if hisia:
      try:
        from hisia import Hisia
        self.hisia = Hisia
      except:
        self.hisia = False
        print('hisia not working')
    else:
      self.hisia=False

  def get_sentiment(self,text,timings=False):
    d = {}
    if text.strip()=='':
        return d
    funcs = [lambda x: self.afinn.score(x),
    lambda text:self.sent.sentida(text,output='total',normal=True,speed ='normal'),
    lambda x: self.classifier.predict(x),
    lambda text: self.classifier_tone.predict(text),
        lambda text: self.hisia(text).sentiment.sentiment,
        lambda text: max(self.nlp(text).cats.items(), key=operator.itemgetter(1))[0]]
    names = ['afinn','sentida','bert_emotion','bert_tone','hisia','spacy_sent']
    d_t = {}
    for func,name in zip(funcs,names):
        try:
            t = time.time()
            d[name] = func(text)
            dt = time.time()-t
            d_t[name] = dt
        except:
            pass
    if timings:
      return d,d_t
    return d
