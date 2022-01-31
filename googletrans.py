### Install the right version of googletrans
dependencies = ['googletrans==3.1.0a0','langdetect','nltk']

for dep in dependencies:
  cmd = 'pip install %s'%dep
  os.system(cmd)
os.system('python -m nltk.downloader punkt')
import time

from langdetect import detect
 ## Import translate package
import googletrans
from googletrans import Translator
translator = Translator() ## Initialize Translator Class
def translate(text,src='da',dest='en'):
  """Function translates text from src language to destination.
  Handles rate limits of google translate."""
  if len(text)>150000: ## check of text is to long, otherwise it should be.
    sents = nltk.sent_tokenize(text)
    segments = []
    temp = [sents[0]] # container for text bits.
    l = len(sents[0]) # counts length of current text bit
    for sent in sents:
      l_t = len(sent)
      if l_t+l > 150000:
        segments.append(' '.join(temp))
      temp = [sent]
      l = len(sent)
    segments.append(' '.join(temp))
    trans = []
    for seg in segments:
      t = translate(seg)
      trans.append(t)
    return ' '.join(trans)


  if detect(text)==dest:
    return text
  for i in range(10):
    res = translator.translate(text,src=src,dest=dest)
    ## Check if translation was successful otherwise wait until limit is reset.
    if res.text == text:
      print('error')
      time.sleep(3600)
    else:
      if i!=0:
        print('success')
      return res.text
