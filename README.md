# Wordmap

The `Wordmap` project is the initial stages of my goal to create a `Knowledge Graph` by parsing unstructured documents.

## Current Approach

`Wordmap` uses the following open-source libraries:
1. For parsing text and word embeddings: [spaCy](https://spacy.io/)
2. For dimensionality reduction: [openTSNE](https://opentsne.readthedocs.io/en/latest/)
3. For k-means clustering: [scikit-learn K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means)

`Wordmap` uses [word embeddings](https://en.wikipedia.org/wiki/Word_embedding) of the unique words in the given text, uses the k-means algorithm to cluster the 300-dimension vectors, reduces the vector dimensions to 2, and then plots the resulting categorized dataset.
