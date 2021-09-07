import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = ['sentida','danlp','allennlp','hisia','afinn']
# installing packages
for package in packages:
    install(package)

import nltk
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
    d_t = {}
    # afinn
    t = time.time()
    if not type(self.afinn)==bool:
      score = self.afinn.score(text)
      d['afinn'] = score
    dt = time.time()-t
    d_t['afinn'] = dt
    t = time.time()
    # sentida
    if not type(self.sent)==bool:
      score = self.sent.sentida(text,output='total',normal=True,speed ='normal')
      d['sentida'] = score
    dt = time.time()-t
    d_t['sentida'] = dt
    t = time.time()
    # bert emotion and tone
    if not type(self.classifier)==bool:
      bert = self.classifier.predict(text)
      d['bert'] = bert
    dt = time.time()-t
    d_t['bert'] = dt
    t = time.time()

    if not type(self.classifier_tone)==bool:
      bert = self.classifier_tone.predict(text)
      d['bert_tone'] = bert
    dt = time.time()-t
    d_t['bert_tone'] = dt
    t = time.time()

    # hisia
    if not type(self.hisia)==bool:
      sent = self.hisia(text)
      sent.sentiment
      d['hisia_sent'] = sent.sentiment
    dt = time.time()-t
    d_t['hisia'] = dt
    t = time.time()

    # spacy model
    if not type(self.nlp)==bool:
      doc = self.nlp(text)
      spac = max(doc.cats.items(), key=operator.itemgetter(1))[0]
      d['spacy_sent'] = spac
    dt = time.time()-t
    d_t['space_sent'] = dt
    t = time.time()
    if timings:
      return d,d_t
    return d
