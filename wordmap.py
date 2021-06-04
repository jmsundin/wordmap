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
    document = nlp(text)
    return [sentence for sentence in document.sents]


def parse_sentences(sentences):
    words = []
    for sentence in sentences:
        for token in sentence:
            if token.is_alpha:
                words.append(token.text)
    
    return words


def get_sentence_vectors(sentences):
    sentence_vecs = []
    for sent in sentences:
        sentence_vecs.append(sent.vector)
    
    return sentence_vecs


def get_word_vectors(nlp, words):
    tokens = [nlp(word) for word in words]
    return np.array([token.vector for token in tokens])


def cosine_similarity(sentence_vec, word_vec):
    return np.dot(sentence_vec, word_vec)/(np.linalg.norm(sentence_vec)*np.linalg.norm(word_vec))

def get_sentence_vec_label(sentence_vecs, word_vecs, sorted_words):
    # Using cosine similarity to determine which word vector is most similar
    # to the sentence vector to then use that word as the label for the
    # sentence vector
    sentence_label_dict = {}
    words_dict = {}
    updated_sentence_vecs_list = [] # list of sentence vectors that are not zero vectors
    updated_words_list = [] # list of words with most similarity to given sentence vectors
    for i, sentence_vec in enumerate(sentence_vecs):
        # np.any() returns true if vector is not the zero vector
        if np.any(sentence_vec):
            for j, word_vec in enumerate(word_vecs):
                if np.any(word_vec):
                    words_dict[j] = cosine_similarity(sentence_vec, word_vec)
            if None not in words_dict:
                word_index = max(words_dict, key=words_dict.get)
                updated_sentence_vecs_list.append(sentence_vecs[i])
                updated_words_list.append(sorted_words[word_index])

    return updated_sentence_vecs_list, updated_words_list


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


def plot(df, updated_words_list, y):
    fig, ax = plt.subplots()
    scatter = ax.scatter(x=df[0], y=df[1], c=y, alpha=0.5)
    ax.grid(color='grey', linestyle='solid')
    ax.set_title('Wordmap')
    labels = updated_words_list
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

    print('Loading spaCy language model')
    start_time_load_model = time.time()
    nlp = spacy.load('en_core_web_md') # English core web medium model
    # nlp = spacy.load('en_core_web_lg') # Load the English core web large model
    end_time_load_model = time.time()
    print('Loading model time: ', end_time_load_model - start_time_load_model)

    print('Getting sentences')
    sentences = get_sentences(nlp, text) # returns a list of sentences with Span type (spacy type)
    print('Sentences MB: ', sys.getsizeof(sentences)/1024)
    sentence_vecs = get_sentence_vectors(sentences)

    print('Parsing sentences')
    words = parse_sentences(sentences) # returns list of Token type
    
    # Create a sorted list of unique words with set()
    sorted_words = sorted(set(words))

    print('Getting word vectors')
    word_vecs = get_word_vectors(nlp, sorted_words)

    updated_sentence_vecs_list, updated_words_list = get_sentence_vec_label(sentence_vecs, word_vecs, sorted_words)

    print(updated_words_list)
    np_array_sent_vecs = np.array(updated_sentence_vecs_list)

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
    df = pd.DataFrame(data=coordinates, index=updated_words_list)
    plot(df, updated_words_list, y)


if __name__ == '__main__':
    main()