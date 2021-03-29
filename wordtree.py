# wordtree
import nltk
nltk.download('punkt')
from IPython.display import Markdown, display
from IPython.core.display import HTML
#Here's the answer to the blankpage issue:
#http://stackoverflow.com/questions/9519673/why-does-google-load-cause-my-page-to-go-blank
from string import Template
import random
from IPython.display import Markdown, display
from IPython.core.display import HTML
#Here's the answer to the blankpage issue:
#http://stackoverflow.com/questions/9519673/why-does-google-load-cause-my-page-to-go-blank
from string import Template
import random
class Wordtree():
    def __init__(self,docs,tokenizer=nltk.word_tokenize):
        self.tokenized_docs = [tokenizer(doc) if type(doc)==str else doc for doc in docs]
        corpus = [word for doc in self.tokenized_docs for word in doc]
        self.corpus = corpus
        w2idx = {w.lower():[] for w in corpus}
        for num,w in enumerate(corpus):
            w2idx[w.lower()].append(num)
        self.w2idx = w2idx
        self.n_corpus = len(corpus)
    def get_concordance(self,word, before=5, after=5,k=20):
        import random
        w = word.lower()

        if w in self.w2idx:

            indices = self.w2idx[w]
            indices = random.sample(indices,min([len(indices),k]))
        else:
            return []
        sentences = [self.corpus[max([i-before,0]):min([i+before,self.n_corpus])] for i in indices]
        #print(len(sentences))
        return sentences
    def printmd(string):
        display(Markdown(string))

    def display_concordance(self,word='we',before=5,after=5,k=20,full=False):





    #myword = "Athens"
    #textList.concordance(word, width=100, lines=40)
    #Turn concordance output into list of lists for WordTree visualisation
    #conc = (cap.stdout).encode('utf-8')
    #conc = conc.decode('utf-8')
    #print(conc)
    #kwic = conc.split('\n')#[1:-1]

        if full: #
            n = len(self.tokenized_docs)
            n_words = sum(map(len,self.tokenized_docs))
            av_words = n_words/n
            sample_size = 10000/av_words
            if n>sample_size:
                sentences = random.sample(self.tokenized_docs,int(sample_size))
            else:
                sentences = self.tokenized_docs
            kwic_data = [[' '.join(words)] for words in sentences]#[[kwic[i]] for i in range(0,len(kwic))]
            kwic_data = [['Phrases']]+kwic_data #replace "Displaying n of n matches' with header for WordTree - 'Phrases'
        else:
            kwic_data = [[' '.join(words)] for words in self.get_concordance(word,before=before,after=after,k=k)]#[[kwic[i]] for i in range(0,len(kwic))]
            kwic_data = [['Phrases']]+kwic_data #replace "Displaying n of n matches' with header for WordTree - 'Phrases'
        print(len(kwic_data))

        js_text_template = Template('''
            <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
            <script type="text/javascript">
              google.charts.load('current', {packages:['wordtree'], callback: function() {
                    // do stuff, if you wan't - it doesn't matter, because the page isn't blank!
                }});
              google.charts.setOnLoadCallback(drawChart);

              function drawChart() {
                var data = google.visualization.arrayToDataTable($kwic_data

                );

                var options = {
                  wordtree: {
                    format: 'implicit',
                    type: 'double',
                    word: '$myword'
                  }
                };

                var chart = new google.visualization.WordTree(document.getElementById('wordtree_basic'));
                chart.draw(data, options);
              }
            </script>

        ''')

        html_text = '''
        <div id="wordtree_basic" style="width: 900px; height: 500px;"></div>
        '''


        js_text = js_text_template.substitute({'kwic_data': kwic_data, 'myword': word})
        #print(js_text)
        #print js_text + html_text
        return HTML(js_text+html_text)
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
def interact_wordtree(wtree,word='we'):
    return interact(wtree.display_concordance, word=word
         ,before=widgets.IntSlider(min=0,max=30,step=1,value=5)
         ,after=widgets.IntSlider(min=0,max=30,step=1,value=5),
        k = widgets.IntSlider(min=0,max=150,step=1,value=25),
        full=False)
# word cloud
