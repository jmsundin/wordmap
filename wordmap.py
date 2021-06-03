import time
import sys
# import subprocess # uncomment if you want to run asciidoc3 on a file
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpld3 # makes the plot interactive

from bs4 import BeautifulSoup
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.matcher import Matcher
from sklearn.cluster import KMeans
from openTSNE import TSNE


def get_sentences(nlp, text):
    config = {'punct_chars': None}
    nlp.add_pipe('sentencizer', config=config)
    
    start_time_create_doc = time.time()
    document = nlp(text)
    end_time_create_doc = time.time()
    print('Amount of time to create doc object from text input: ', end_time_create_doc - start_time_create_doc)
    
    return [sentence for sentence in document.sents]


def parse_sentences(nlp, sentences):
    words = []
    for sentence in sentences:
        for word in sentence.split():
            token = nlp(word)[0] # nlp(word) returns a Doc object. Then access the first token.
            if token.is_alpha:
                words.append(token.text)
    
    return words


def get_sentence_vectors(sentences):
    for sent in sentences:
        sentence_vecs.append(sent.vector)
    return sentence_vecs


def get_word_vectors(nlp, words):
    tokens = [nlp(word) for word in words]
    return [token.vector for token in tokens]


def dimension_reduction(vectors):
    tsne = TSNE(
        perplexity=50, # can be thought of as the continuous k number of nearest neighbors
        #metric='cosine',
        verbose=True,
        n_jobs=-1, # tsne uses all processors; -2 will use all but one processor
        random_state=42, # int used as seed for random number generator
        dof=0.5 # degrees of freedom
    )

    return tsne.fit(vectors) # returns word vectors/embeddings


def clustering(embeddings):
    kmeans = KMeans(n_clusters=8)
    kmeans.fit(embeddings)
    print('KMeans trained')

    # y = kmeans.predict(df.values)
    y = kmeans.predict(embeddings)
    print('KMeans clustering complete')
    
    return y 


def plot(df, sorted_words, y):
    fig, ax = plt.subplots()
    scatter = ax.scatter(x=df[0], y=df[1], c=y, alpha=0.5)
    ax.grid(color='grey', linestyle='solid')
    ax.set_title('Wordmap')
    labels = sorted_words
    tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
    mpld3.plugins.connect(fig, tooltip)

    # Use this method in jupyter notebook
    #mpld3.display()

    # Use this method if you are not using jupyter notebook
    mpld3.show() 


def main():
    # print('Running asciidoc3')
    # try:
    #     subprocess.run(args=['asciidoc3', '-a', 'toc', '-n', '-a', 'icons', 'Chapter 06 -- Reasoning with Word Vectors (Word2vec).adoc'])
    # except SubprocessError as e:
    #     print('Error trying to run asciidoc3: ', e)


    chapter6_html = open('Chapter 06 -- Reasoning with Word Vectors (Word2vec).html', 'r').read()
    bsoup = BeautifulSoup(chapter6_html, 'html.parser')
    text = bsoup.get_text()

    nlp = spacy.load('en_core_web_md') # English core web medium model

    print('Loading spaCy language model')
    start_time_load_model = time.time()
    nlp = spacy.load('en_core_web_lg') # Load the English core web large model
    end_time_load_model = time.time()
    print('Loading English large model time: ', end_time_load_model - start_time_load_model)


    print('Getting sentences')
    start_time_get_sentences = time.time()
    sentences = get_sentences(nlp, text) # returns a list of sentences with span type (spacy type)
    end_time_get_sentences = time.time()
    print('Time for get_sentences function: ', end_time_get_sentences - start_time_get_sentences)
    print('Sentences MB: ', sys.getsizeof(sentences)/1024)
    
    sentence_vecs = get_sentence_vectors(sentences)

    # print('Parsing sentences')
    # start_time_parsing = time.time()
    # words = parse_sentences(nlp, sentences) # returns list
    # end_time_parsing = time.time()
    # print('Time for parsing sentences: ', end_time_parsing - start_time_parsing)

    # # Create a sorted list of unique words with set()
    # sorted_words = sorted(set(words))

    # print('Getting word vectors')
    # start_time_get_vectors = time.time()
    # vectors = get_word_vectors(nlp, sorted_words)
    # end_time_get_vectors = time.time()
    # print('Time to get vectors: ', end_time_get_vectors - start_time_get_vectors)

    # type cast the vector list to a numpy array to use in the DataFrame
    # np_array_vectors = np.array(vectors)

    np_array_sent_vecs = np.array(sentence_vecs)

    print('Starting TSNE')
    start_time_tsne = time.time()
    embeddings = dimension_reduction(np_array_sent_vecs)
    end_time_tsne = time.time()
    print('Time for TSNE: ', end_time_tsne - start_time_tsne)

    print('Starting KMeans')
    start_time_clustering = time.time()
    y = clustering(embeddings)
    end_time_clustering = time.time()
    print('Time for KMeans: ', end_time_clustering - start_time_clustering)

    # Making the data pretty
    coordinates = np.tanh(0.666*embeddings/np.std(embeddings))

    print('Plotting word vectors')
    plot(pd.DataFrame(coordinates, index=sorted_words), sorted_words, y)


if __name__ == '__main__':
    main()