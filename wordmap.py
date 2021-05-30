import time
import sys
import spacy # for parsing text and word embeddings
import pandas as pd # for DataFrame data structure
import numpy as np # for numpy.array data structure
import networkx as nx # Graph library
import matplotlib.pyplot as plt # for scatter plot
import mpld3 # adds interactivity to the scatter plot

from sklearn.cluster import KMeans # unsupervised learning K-means model to cluster word embeddings
from openTSNE import TSNE # for dimensionality reduction of the 300 dimension word embeddings to 2 dimensions
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL # for sentence parsing
from spacy.matcher import Matcher


def get_sentences(nlp, text):
    config = {"punct_chars": None}
    nlp.add_pipe("sentencizer", config=config)
    
    start_time_create_doc = time.time()
    
    document = nlp(text)
    
    end_time_create_doc = time.time()
    
    print("Amount of time to create doc object from text input: ", end_time_create_doc - start_time_create_doc)
    
    return [str(sentence).strip() for sentence in document.sents]


def parse_sentence(nlp, sentences):
    head = ''
    relation = ''
    tail = ''

    parser_obj = nlp.add_pipe("parser")
    parsed = parser_obj(sentences)
    for sentence in sentences:
        pass
    return []


def show_graph():
    pass


def knowledge_graph():
    pass
    # dependencies = parse_sentence(nlp, sentences)


    # words = [] # strings of the tokens

    # for token in doc:
    #     if token.is_alpha:
    #         words.append(token.text)

    # Create a sorted list of unique words with set()
    # words = sorted(set(words))
    # print(len(words))
    # print('Create a sorted list of unique words with set()')


    # Create Token objects of the words list to create word vectors/embeddings
    # tokens = []
    # for word in words:
    #     tokens.append(nlp(word)[0])
        
    # print(len(tokens))
    # print('Create Token objects of the words list to create word vectors/embeddings')

    # Create word vectors/embeddings from the Token objects
    # vectors = []
    # for token in tokens:
    #     vectors.append(token.vector)

    # print('Create word vectors/embeddings from the Token objects')


    # cast the vector list to a numpy array to use in the DataFrame
    # vectors = np.array(vectors)


    # dimensionality reduction of the word vectors/embeddings
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
    # English core web medium model
    # nlp = spacy.load('en_core_web_md')

    start_time_load_model = time.time()

    # Load the English core web large model
    nlp = spacy.load('en_core_web_lg')

    end_time_load_model = time.time()
    
    print("Loading English large model time: ", end_time_load_model - start_time_load_model)

    # open and read ascii document
    text = open('Chapter 01 -- Packets of Thought (NLP Overview).asc', 'r').read()

    start_time_get_sentences = time.time()

    # returns a list of sentences with str type
    sentences = get_sentences(nlp, text)

    end_time_get_sentences = time.time()

    print("Time for get_sentences function: ", end_time_get_sentences - start_time_get_sentences)
    print("Sentences MB: ", sys.getsizeof(sentences)/1024)

    # knowledge_graph()