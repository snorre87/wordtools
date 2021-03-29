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
            sample_size = 50000/av_words
            print(sample_size)
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
    def interact_wordtree(self,word='we'):
        return interact(self.display_concordance, word=word
             ,before=widgets.IntSlider(min=0,max=30,step=1,value=5)
             ,after=widgets.IntSlider(min=0,max=30,step=1,value=5),
            k = widgets.IntSlider(min=0,max=150,step=1,value=25),
            full=False)
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

# word cloud
from wordcloud import (WordCloud, get_single_color_func)
import matplotlib.pyplot as plt


class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


class GroupedColorFunc(object):
    """Create a color function object which assigns DIFFERENT SHADES of
       specified colors to certain words based on the color to words mapping.

       Uses wordcloud.get_single_color_func

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color='red'):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)
import matplotlib
def color_by_groups(wc,w2group,cmap = plt.cm.tab20,default_color='grey'):
    groups = set(w2group.values())
    g2color = {}
    for num,g in enumerate(groups):
        color = matplotlib.colors.rgb2hex( cmap(num/len(groups)))
    #    print(color)
        g2color[g] = color
    color_to_words = {color:[] for g,color in g2color.items()}
    for w,g in w2group.items():
        color = g2color[g]
        color_to_words[color].append(w)
    grouped_color_func = GroupedColorFunc(color_to_words, default_color)
# Apply our color function
    wc.recolor(color_func=grouped_color_func)
    # Plot
    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    return wc
def get_wordclod(word_freq=False,text=False,collocations=False):
    if text!=False:
        wc = WordCloud(collocations=False).generate(text.lower())
    if word_freq!=False:
        wc = WordCloud(collocations=False).generate_from_frequencies(word_freq)
    return wc
