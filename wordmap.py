import time
import sys
import os.path

# import subprocess # uncomment if you want to run asciidoc3 on a file
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


def read_write(file_name: str, mode: str, items: list = []):
    with open(file_name, mode) as f:
        if mode == 'wb':
            np.save(f, items, allow_pickle=True)
            return None            
        elif mode == 'rb':
            items = np.load(f, allow_pickle=True)
            return items


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
        sent_vec = sent.vector
        if not (0.0 in sent_vec or 0. in sent_vec):
            sent_vecs.append(sent_vec)
    return sent_vecs


def get_word_vecs(nlp: spacy.language.Language, words: list) -> np.array:
    tokens = [nlp(word) for word in words]
    return np.array([token.vector for token in tokens])


def get_sent_vec_norm(sent_vecs: list) -> np.array:
    for i, sent_vec in enumerate(sent_vecs):
        sent_vecs[i] = sent_vec / np.linalg.norm(sent_vec) # 2-norm is default with vectors
    return np.array(sent_vecs)


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


def create_adj_graph(similarity_matrix: np.ndarray, G: nx.Graph) -> None:
    similarity_matrix = np.triu(similarity_matrix, k=1)
    it = np.nditer(similarity_matrix, flags=['multi_index'], order='C')
    for edge in it:
        if edge > 0.95:
            # print(f'Adding edge: ({it.multi_index[0]}, {it.multi_index[1]}, weight = {edge})')
            G.add_edge(it.multi_index[0], it.multi_index[1], weight=edge)
    
    # Other attempts:

    # Diagonal indices
    # indices = np.diag_indices_from(similarity_matrix)

    # for diag_i, diag_j in zip(indices[0], indices[1]):
    #     for index, value in enumerate(similarity_matrix[diag_i, diag_j + 1])
    #         if j > 0.95:
    #             print(f'Adding edge ({i}, {j})')
    #             G.add_edge(i, j)   
    #     diagonal += 1

    # n = len(similarity_matrix[0])
    # upper_triangle_indices_no_diag = np.triu_indices(n-1, k=1) # n-1 is the size because the shift from the diag is 1 (k=1)
    # print(f'Upper indices[0]: {upper_triangle_indices_no_diag[0]}')
    # print(f'Upper indices[1]: {upper_triangle_indices_no_diag[1]}')
    # first_loop = 0
    # loop through indices of the upper triangle from similarity_matrix and add [i, j] to adjacency graph
    # for i, index_i in enumerate(upper_triangle_indices_no_diag[0]):
    #     first_loop += 1
    #     print(f'{first_loop = }')
    #     for j, index_j in enumerate(upper_triangle_indices_no_diag[1]):
    #         # print(f'{j = }')
    #         if similarity_matrix[index_i, index_j] > 0.95:
    #             G.add_edge(index_i, index_j, similarity_matrix[index_i][index_j])
    #             print(f'Added edge: ({index_i}, {index_j}); Value: {similarity_matrix[index_i, index_j]}')
    #         else:
    #             continue


def plot_adj_graph(G: nx.Graph) -> None:
    plt.subplot(1, 1, 1)
    pos = nx.spring_layout(G, k=0.15, seed=42)
    nx.draw_networkx(G, pos=pos)
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
    sent_vecs_npy = 'sentence_vectors.npy'

    if not os.path.exists(chap6_word_vecs_html):
        print('Running asciidoc3')
        try:
            subprocess.run(args=['asciidoc3', '-a', 'toc', '-n', '-a', 'icons', 'Chapter 06 -- Reasoning with Word Vectors (Word2vec).adoc'])
        except SubprocessError as e:
            print('Error trying to run asciidoc3: ', e)
    
    if os.path.exists(chap6_word_vecs_html):
        mode_r = 'r'
        chapter6_html = open(chap6_word_vecs_html, mode_r).read()
        bsoup = BeautifulSoup(chapter6_html, 'html.parser')
        text = bsoup.get_text()

    if os.path.exists(sent_vecs_npy):
        # reading sentence vectors from file
        mode_r = 'rb'
        np_array_sent_vecs_norm = read_write(sent_vecs_npy, mode_r)
        # print('sentence_vectors.npy: ', np_array_sent_vecs_norm[:10])
        print('Number of sentence vectors: ', len(np_array_sent_vecs_norm))

    if not os.path.exists(sent_vecs_npy):
        print('Loading spaCy language model')
        start_time_load_model = time.time()
        nlp = spacy.load('en_core_web_md') # English core web medium model
        # nlp = spacy.load('en_core_web_lg') # Load the English core web large model
        end_time_load_model = time.time()
        print('Loading model time: ', end_time_load_model - start_time_load_model)
        
        print('Getting sentences')
        sentences = get_sents(nlp, text) # returns a list of sentences with Span type (spacy type)
        print('Number of sentences: ', len(sentences))
        print('Sentences MB: ', sys.getsizeof(sentences)/1024)

        np_array_sent_vecs_norm = get_sent_vec_norm(get_sent_vecs(sentences))

        # writing sentence vectors to file
        mode_w = 'wb'
        read_write(sent_vecs_npy, mode_w, np_array_sent_vecs_norm)

    # print('Parsing sentences')
    # words = parse_sents(sentences) # returns list of Token type
    
    # # Create a sorted list of unique words with set()
    # sorted_words = sorted(set(words))

    # print('Getting word vectors')
    # word_vecs = get_word_vecs(nlp, sorted_words)

    # updated_sent_vecs_list, updated_words_list = get_sent_vec_label(sent_vecs, word_vecs, sorted_words)

    # print(updated_words_list)
  
    similarity_matrix = get_similarity_matrix(np_array_sent_vecs_norm)

    # Creating the adjacency matrix
    # First implementation:
    # adjacency_matrix = similarity_matrix > 0.997 # returning True for every entry that is greater than 0.997
    # adjacency_matrix = adjacency_matrix.astype(int) # changing True to 1 and False to 0
    # Second implementation:
    G = nx.Graph()
    create_adj_graph(similarity_matrix, G)
    # print(G.edges)
    # print(G.adj)
    plot_adj_graph(G)
  
    ### TODO:
    # pd.df.corr()
    # largest noun phrases for labels


    # print('Starting TSNE')
    # start_time_tsne = time.time()
    # embeddings = dim_red(np_array_sent_vecs)
    # end_time_tsne = time.time()
    # print('Time for TSNE: ', end_time_tsne - start_time_tsne)

    # print('Starting KMeans')
    # start_time_clustering = time.time()
    # cluster_array = clustering(embeddings)
    # end_time_clustering = time.time()
    # print('Time for KMeans: ', end_time_clustering - start_time_clustering)

    # # Making the data pretty
    # coordinates = np.tanh(0.666*embeddings/np.std(embeddings))

    # print('Plotting word vectors')
    # df = pd.DataFrame(data=coordinates, index=updated_words_list)
    # plot_word_cloud(df, updated_words_list, cluster_array)


if __name__ == '__main__':
    main()