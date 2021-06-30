import time
import sys
import os.path

import subprocess
import spacy
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import mpld3 # makes the plot interactive

from bs4 import BeautifulSoup
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.matcher import Matcher
from sklearn.cluster import KMeans
from openTSNE import TSNE
from collections import Counter


def read_write(file_name: str, mode: str, nlp=None, doc=None, items: list=None):
    with open(file_name, mode) as f:
        if mode == 'wb':
            if nlp is not None:
                nlp_bytes_data = nlp.to_bytes()
                f.write(nlp_bytes_data)
            if doc is not None:
                f.write(doc.to_bytes())
            if '.txt' in file_name and items is not None:
                f.write(bytes(str(items), encoding='utf-8'))
            if '.npy' in file_name:
                np.save(f, items, allow_pickle=True)
                return None            
        elif mode == 'rb':
            if 'nlp' in file_name:
                lang_cls = spacy.util.get_lang_class(config["nlp"]["lang"])
                nlp = lang_cls.from_config(config)
                return nlp.from_bytes(f.read())
            if 'doc' in file_name:
                from spacy.tokens import Doc
                from spacy.vocab import Vocab
                doc_bytes = bytes(f.read())
                return Doc(Vocab()).from_bytes(doc_bytes)
            if 'noun' in file_name:
                noun_phrases = f.readlines()
                return noun_phrases
            if '.npy' in file_name:
                items = np.load(f, allow_pickle=True)
                return items


def get_word_vecs(nlp: spacy.language.Language, words: list) -> np.array:
    tokens = [nlp(word) for word in words]
    return np.array([token.vector for token in tokens])


def get_sents_and_noun_phrases(doc: spacy.tokens.doc.Doc) -> list:
    sentences = []
    noun_phrases = []
    for sent in doc.sents:
        sent_noun_chunks = list(sent.noun_chunks)
        if sent_noun_chunks:
            sentences.append(sent)
            noun_phrases.append(max(sent_noun_chunks))
    print(f'Noun_phrases len: {len(noun_phrases)}')
    return sentences, noun_phrases


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

    print(f'Sent_vecs len: {len(sent_vecs)}')
    return sent_vecs


def get_sent_vec_norm(sent_vecs: list) -> np.array:
    for i, sent_vec in enumerate(sent_vecs):
        sent_vecs[i] = sent_vec / np.linalg.norm(sent_vec) # 2-norm is default with vectors
    return np.array(sent_vecs)


def get_most_common_noun_phrases(doc: spacy.tokens.doc.Doc) -> dict:
    return Counter(list(doc.noun_chunks)).most_common(5)


def get_similarity_matrix(np_array_sent_vecs_norm: np.array) -> list:
    return np_array_sent_vecs_norm.dot(np_array_sent_vecs_norm.T)


def cos_similarity(sent_vec: list, word_vec: list) -> int:
    return np.dot(sent_vec, word_vec)/(np.linalg.norm(sent_vec)*np.linalg.norm(word_vec))


def get_sent_vec_label(sent_vecs: list, word_vecs: list, sorted_words: list) -> list:
    # Using cosine similarity to determine which word vector is most similar
    # to the sentence vector to then use that word as the label for the
    # sentence vector
    sent_label_dict = {}
    words_dict = {}
    updated_sent_vecs_list = [] # list of sentence vectors that are not zero vectors
    updated_words_list = [] # list of words with most similarity to given sentence vectors
    for i, sent_vec in enumerate(sent_vecs):
        # np.any() returns true if vector is not the zero vector
        if np.any(sent_vec):
            for j, word_vec in enumerate(word_vecs):
                if np.any(word_vec):
                    words_dict[j] = cos_similarity(sent_vec, word_vec)
            if None not in words_dict:
                word_index = max(words_dict, key=words_dict.get)
                updated_sent_vecs_list.append(sent_vecs[i])
                updated_words_list.append(sorted_words[word_index])
    return updated_sent_vecs_list, updated_words_list


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


def create_adj_graph(similarity_matrix: np.ndarray, G: nx.Graph, noun_phrases: list) -> dict:
    similarity_matrix = np.triu(similarity_matrix, k=1)
    iterator = np.nditer(similarity_matrix, flags=['multi_index'], order='C')
    node_labels = dict()
    for edge in iterator:
        key = 0
        value = ''
        if edge > 0.95:
            key = iterator.multi_index[0]
            value = noun_phrases[iterator.multi_index[0]]
            # if str(value).isalnum():
            node_labels[key] = value
            G.add_node(iterator.multi_index[0])
            # print(f'Noun_phrase: {noun_phrases[iterator.multi_index[0]]}')
            G.add_edge(iterator.multi_index[0], iterator.multi_index[1], weight=edge)
            # print(f'Adding edge: ({iterator.multi_index[0]}, {iterator.multi_index[1]}, weight = {edge})')
    return node_labels


def plot_adj_graph(G: nx.Graph, node_labels: dict) -> None:
    plt.subplot(1, 1, 1)
    pos = nx.spring_layout(G, k=0.15, seed=42)
    nx.draw_networkx(G, pos=pos, with_labels=True, labels=node_labels, font_weight='bold')
    plt.show()


def plot_word_cloud(df: pd.DataFrame, updated_words_list: list, cluster_array: np.ndarray) -> None:
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
    chap6_word_vecs_html = 'Chapter 06 -- Reasoning with Word Vectors (Word2vec).html'
    spacy_doc_txt = 'spacy_doc.txt'
    sent_vecs_npy = 'sentence_vectors.npy'
    noun_phrases_txt = 'noun_phrases.txt'
    mode_rb = 'rb'
    mode_wb = 'wb'
    mode_r = 'r'
    mode_w = 'w'

    if not os.path.exists(chap6_word_vecs_html) or os.path.getsize(chap6_word_vecs_html) == 0:
        print('Running asciidoc3')
        try:
            subprocess.run(args=['asciidoc3', '-a', 'toc', '-n', '-a', 'icons', 'Chapter 06 -- Reasoning with Word Vectors (Word2vec).adoc'])
        except SubprocessError as e:
            print('Error trying to run asciidoc3: ', e)
    
    if os.path.exists(chap6_word_vecs_html) and os.path.getsize(chap6_word_vecs_html) > 0:
        chapter6_html = open(chap6_word_vecs_html, mode_r).read()
        bsoup = BeautifulSoup(chapter6_html, 'html.parser')
        text = bsoup.get_text()
    
    print('Loading spaCy language model')
    start_time_load_model = time.time()
    nlp = spacy.load('en_core_web_md') # English core web medium model
    # nlp = spacy.load('en_core_web_lg') # Load the English core web large model
    end_time_load_model = time.time()
    print('Loading model time: ', end_time_load_model - start_time_load_model)

    config = {'punct_chars': None}
    nlp.add_pipe('sentencizer', config=config)
    doc = nlp(text)

    if os.path.exists(sent_vecs_npy) and os.path.getsize(sent_vecs_npy) > 0:
        # reading sentence vectors from file
        np_array_sent_vecs_norm = read_write(sent_vecs_npy, mode_rb)
        noun_phrases = read_write(noun_phrases_txt, mode_rb)
        # print('sentence_vectors.npy: ', np_array_sent_vecs_norm[:10])
        print('Number of sentence vectors: ', len(np_array_sent_vecs_norm))

    if not os.path.exists(sent_vecs_npy) or os.path.getsize(sent_vecs_npy) == 0:
        print('Getting sentences')
        # '0' in noun_phrases_0 -> del noun_phrases assoc w/ zero vectors
        sentences, noun_phrases = get_sents_and_noun_phrases(doc)
        
        print('Number of sentences: ', len(sentences))
        print('Sentences MB: ', sys.getsizeof(sentences)/1024)

        sent_vecs = get_sent_vecs(sentences)
        np_array_sent_vecs_norm = get_sent_vec_norm(sent_vecs)

        # writing sentence vectors to file
        read_write(sent_vecs_npy, mode_w, items=np_array_sent_vecs_norm)
        read_write(noun_phrases_txt, mode_wb, items=noun_phrases)

    similarity_matrix = get_similarity_matrix(np_array_sent_vecs_norm)
    G = nx.Graph()
    node_labels = create_adj_graph(similarity_matrix, G, noun_phrases)
    plot_adj_graph(G, node_labels)

    # embeddings = dim_red(np_array_sent_vecs)
    # cluster_array = clustering(embeddings)
    # # Making the data pretty
    # coordinates = np.tanh(0.666*embeddings/np.std(embeddings))
    # df = pd.DataFrame(data=coordinates, index=updated_words_list)
    # plot_word_cloud(df, updated_words_list, cluster_array)


if __name__ == '__main__':
    main()