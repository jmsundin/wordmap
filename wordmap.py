import time
import sys
import spacy # for parsing text and word embeddings
import pandas as pd # for DataFrame data structure
import numpy as np # for numpy.array data structure
import matplotlib.pyplot as plt # for scatter plot
import mpld3 # adds interactivity to the scatter plot

from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL # for sentence parsing
from spacy.matcher import Matcher
from sklearn.cluster import KMeans # unsupervised learning K-means model to cluster word embeddings
from openTSNE import TSNE # for dimensionality reduction of the 300 dimension word embeddings to 2 dimensions


def get_sentences(nlp, text):
    config = {"punct_chars": None}
    nlp.add_pipe("sentencizer", config=config)
    
    start_time_create_doc = time.time()
    
    document = nlp(text)
    
    end_time_create_doc = time.time()
    
    print("Amount of time to create doc object from text input: ", end_time_create_doc - start_time_create_doc)
    
    return [str(sentence).strip() for sentence in document.sents]


def parse_sentences(nlp, sentences):
    words = []
    for sentence in sentences:
        for word in sentence.split():
            token = nlp(word)[0] # nlp(word) returns a Doc object. Then access the first token.
            if token.is_alpha:
                words.append(token.text)
    
    return words


def get_word_vectors(nlp, words):
    tokens = [nlp(word) for word in words]
    return [token.vector for token in tokens]

def dimension_reduction(vectors):
    tsne = TSNE(
        perplexity=50, # can be thought of as the continuous k number of nearest neighbors
        #metric="cosine",
        verbose=True,
        n_jobs=-1, # tsne uses all processors; -2 will use all but one processor
        random_state=42, # int used as seed for random number generator
        dof=0.5 # degrees of freedom
    )

    return tsne.fit(vectors) # returns word vectors/embeddings


def clustering(embeddings):
    kmeans = KMeans(n_clusters=8)
    kmeans.fit(embeddings)
    print('kmeans algorithm trained')

    # y = kmeans.predict(df.values)
    y = kmeans.predict(embeddings)
    print('kmeans classification made')
    
    return y


def plot(df, y):
    fig, ax = plt.subplots()
    scatter = ax.scatter(x=df[0], y=df[1], c=y, alpha=0.5)
    ax.grid(color='grey', linestyle='solid')
    ax.set_title("Wordmap")
    labels = words
    tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
    mpld3.plugins.connect(fig, tooltip)

    # Use this method in jupyter notebook
    #mpld3.display()

    # Use this method if you are not using jupyter notebook
    mpld3.show() 


def main():
    # nlp = spacy.load('en_core_web_md') # English core web medium model

    start_time_load_model = time.time()
    nlp = spacy.load('en_core_web_lg') # Load the English core web large model
    end_time_load_model = time.time()
    print("Loading English large model time: ", end_time_load_model - start_time_load_model)

    # document
    text = open('Chapter 01 -- Packets of Thought (NLP Overview).asc', 'r').read()

    start_time_get_sentences = time.time()
    sentences = get_sentences(nlp, text) # returns a list of sentences with str type
    end_time_get_sentences = time.time()
    print("Time for get_sentences function: ", end_time_get_sentences - start_time_get_sentences)
    print("Sentences MB: ", sys.getsizeof(sentences)/1024)
    
    words = parse_sentences(nlp, sentences) # returns list
    
    # Create a sorted list of unique words with set()
    words = sorted(set(words))

    vectors = get_word_vectors(nlp, words)

    # type cast the vector list to a numpy array to use in the DataFrame
    np_array_vectors = np.array(vectors)


    embeddings = dimension_reduction(np_array_vectors)
    
    y = clustering(embeddings)

    # Making the data pretty
    coordinates = np.tanh(0.666*embeddings/np.std(embeddings))

    plot(pd.DataFrame(coordinates, index=words), y)


if __name__ == '__main__':
    main()