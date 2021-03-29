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
def color_by_groups(wc,w2group,cmap = plt.cm.tab20,default_color='grey'):
    groups = set(w2group.items())
    g2color = {}
    for num,g in enumerate(groups):
        g2color[g] = cmap(num/len(groups))
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
