# Wordmap Current Status:
# - Graph of clustered word embeddings with hoverover text popup
# - Using spaCy to parse and tokenize text.
# - Using spaCy to get word vectors/embeddings for each of the unique words in my text.
# - Clustering is performed on the word vectors using Kmeans clustering to "classify" the words.
#     - "Similar" words have the same color.

# **spaCy api -- Architecture**
# - [spaCy api link](https://spacy.io/api)


import spacy # for parsing text and word embeddings
import pandas as pd # for DataFrame data structure
import numpy as np # for numpy.array data structure
import networkx as nx # Graph library
import matplotlib.pyplot as plt # for scatter plot
import mpld3 # adds interactivity to the scatter plot
from sklearn.cluster import KMeans # unsupervised learning K-means model to cluster word embeddings
from openTSNE import TSNE # for dimensionality reduction of the 300 dimension word embeddings to 2 dimensions
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL # for sentence parsing



def get_sentences(nlp, text):
    config = {"punct_chars": None}
    nlp.add_pipe("sentencizer", config=config)
    
    return nlp(text)


def parse_sentence(sentences):
    for sentence in sentences:


def show_graph():
    pass

def wordmap():

    # English core web medium model
    # nlp = spacy.load('en_core_web_md')

    nlp = spacy.load('en_core_web_lg')

    # text = open('Chapter 01 -- Packets of Thought (NLP Overview).asc', 'r').read()

    text = '''You are about to embark on an exciting adventure in natural language processing (NLP).
First we show you what NLP is and all the things you can do with it.
This will get your wheels turning, helping you think of ways to use NLP in your own life both at work and at home.

Then we dig into the details of exactly how to process a small bit of English text using a programming language like Python, which will help you build up your NLP toolbox incrementally.
In this chapter you'll write your first program that can read and write English statements.
This Python snippet will be the first of many you'll use to learn all the tricks needed to assemble an English language dialog engine -- a chatbot.

== Natural language vs. programming language'''

    sentences = list(get_sentences(nlp, text).sents)
    
    parse_sentence(sentences)


    # words = [] # strings of the tokens

    # for token in doc:
    #     if token.is_alpha:
    #         words.append(token.text)

    # # Create a sorted list of unique words with set()
    # words = sorted(set(words))
    # print(len(words))
    # print('Create a sorted list of unique words with set()')


    # # Create Token objects of the words list to create word vectors/embeddings
    # tokens = []
    # for word in words:
    #     tokens.append(nlp(word)[0])
        
    # print(len(tokens))
    # print('Create Token objects of the words list to create word vectors/embeddings')

    # # Create word vectors/embeddings from the Token objects
    # vectors = []
    # for token in tokens:
    #     vectors.append(token.vector)

    # print('Create word vectors/embeddings from the Token objects')


    # # cast the vector list to a numpy array to use in the DataFrame
    # vectors = np.array(vectors)


    # # dimensionality reduction of the word vectors/embeddings
    # tsne = TSNE(
    #     perplexity=50,
    #     #metric="cosine",
    #     verbose=True,
    #     n_jobs=-2,
    #     random_state=42,
    #     dof=0.5
    # )
    # print('tsne object created')

    # embeddings = tsne.fit(vectors)
    # print('Embeddings: ', len(embeddings))
    # print('Words: ', len(words))

    # kmeans = KMeans(n_clusters=8)
    # kmeans.fit(embeddings)
    # print('kmeans algorithm trained')

    # # y = kmeans.predict(df.values)
    # y = kmeans.predict(embeddings)
    # print('kmeans classification made')

    # # Making the data pretty
    # coordinates = np.tanh(0.666*embeddings/np.std(embeddings))


    # df = pd.DataFrame(coordinates, index=words)
    # #df

    # # Scatter plot
    # fig, ax = plt.subplots()
    # scatter = ax.scatter(x=df[0], y=df[1], c=y, alpha=0.5)
    # ax.grid(color='grey', linestyle='solid')
    # ax.set_title("Wordmap")
    # labels = words
    # tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
    # mpld3.plugins.connect(fig, tooltip)

    # # Use this method in jupyter notebook
    # #mpld3.display()

    # # Use this method if you are not using jupyter notebook
    # mpld3.show() 


if __name__ == '__main__':
    wordmap()