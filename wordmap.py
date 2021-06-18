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


def get_sents(nlp: spacy.language.Language, text: str) -> list:
    config = {'punct_chars': None}
    nlp.add_pipe('sentencizer', config=config)
    document = nlp(text)
    return [sentence for sentence in document.sents]


def parse_sents(sentences: list) -> list:
    words = []
    for sentence in sentences:
        for token in sentence:
            if token.is_alpha:
                words.append(token.text)
    
    return words


def get_sent_vecs(sentences: list) -> list:
    sent_vecs = []
    for sent in sentences:
        sent_vecs.append(sent.vector)
    
    return sent_vecs


def get_word_vecs(nlp: spacy.language.Language, words: list) -> np.array:
    tokens = [nlp(word) for word in words]
    return np.array([token.vector for token in tokens])


def get_vec_norm(sent_vecs: list) -> np.array:
    for i, sent in enumerate(sent_vecs):
        sent_vecs[i] = np.linalg.norm(sent)

    return np.array(sent_vecs)


def cos_similarity(sent_vec: list, word_vec: list) -> int:
    return np.dot(sent_vec, word_vec)/(np.linalg.norm(sent_vec)*np.linalg.norm(word_vec))

def get_sent_vec_label(sent_vecs: list, word_vecs: list, sorted_words: list) -> list:
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


def dim_reduc(vecs: list) -> list:
    tsne = TSNE(
        perplexity=50, # can be thought of as the continuous k number of nearest neighbors
        #metric='cosine',
        verbose=True,
        n_jobs=-1, # tsne uses all processors; -2 will use all but one processor
        random_state=42, # int used as seed for random number generator
        dof=0.5 # degrees of freedom
    )
    return tsne.fit(vectors) # returns word vectors/embeddings


def clustering(embeddings: list) -> np.ndarray:
    kmeans = KMeans(n_clusters=8)
    kmeans.fit(embeddings)
    print('KMeans trained')

    # cluster_array = kmeans.predict(df.values)
    cluster_array = kmeans.predict(embeddings)
    print('KMeans clustering complete')
    return cluster_array


def plot(df: pd.DataFrame, updated_words_list: list, cluster_array: np.ndarray) -> None:
    fig, ax = plt.subplots()
    scatter = ax.scatter(x=df[0], y=df[1], c=cluster_array, alpha=0.5)
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
    sentences = get_sents(nlp, text) # returns a list of sentences with Span type (spacy type)
    print('Sentences MB: ', sys.getsizeof(sentences)/1024)
    sent_vecs = get_sent_vecs(sentences)

    print('Parsing sentences')
    words = parse_sents(sentences) # returns list of Token type
    
    # Create a sorted list of unique words with set()
    sorted_words = sorted(set(words))

    print('Getting word vectors')
    word_vecs = get_word_vecs(nlp, sorted_words)

    updated_sent_vecs_list, updated_words_list = get_sent_vec_label(sent_vecs, word_vecs, sorted_words)

    print(updated_words_list)
    np_array_sent_vecs = np.array(updated_sent_vecs_list)

    # create n by 300 normalized sent_vec
    np_array_norm_sent_vecs = get_sent_vecs(sent_vecs)
    
    similarity_matrix = get_similarity_matrix(np_array_norm_sent_vecs)
    
    # similarity_matrix = sent_vec.dot(sent_vec.T) = similarity matrix
    # assert similarity_matrix.diag == 1 -> .diag returns matrix of diagonal
    # thresholds -> .9 or .8 -> create new tuple of all rows that pass this
    # sklearn -> unsupervised learning clustering


    print('Starting TSNE')
    start_time_tsne = time.time()
    embeddings = dim_red(np_array_sent_vecs)
    end_time_tsne = time.time()
    print('Time for TSNE: ', end_time_tsne - start_time_tsne)

    print('Starting KMeans')
    start_time_clustering = time.time()
    cluster_array = clustering(embeddings)
    end_time_clustering = time.time()
    print('Time for KMeans: ', end_time_clustering - start_time_clustering)

    # Making the data pretty
    coordinates = np.tanh(0.666*embeddings/np.std(embeddings))

    print('Plotting word vectors')
    df = pd.DataFrame(data=coordinates, index=updated_words_list)
    plot(df, updated_words_list, cluster_array)


if __name__ == '__main__':
    main()