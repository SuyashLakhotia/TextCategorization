# Text Categorization

> This repository contains the source code and other helper files for my undergraduate thesis titled "Graph Convolutional Neural Networks for Text Categorization" under the supervision of [Prof. Xavier Bresson](http://www.ntu.edu.sg/home/xbresson/) at Nanyang Technological University, Singapore.

There are a total of three benchmark models and three deep learning models implemented in this repository for text classification:

1. `baseline.py`: Linear SVC & Multinomial Naive Bayes
2. `mlp.py`: Multilayer Perceptron
3. `cnn_fchollet.py`: F. Chollet CNN (based on this [2016 blog post](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html))
4. `cnn_ykim.py`: Y. Kim CNN (based on [Y. Kim, 2014](https://arxiv.org/abs/1408.5882))
5. `graph_cnn.py`: Graph CNN (based on [M. Defferrard et al., 2017](https://arxiv.org/abs/1606.09375))

The above models were tested on three datasets &mdash; [Rotten Tomatoes Sentence Polarity Dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/), [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) & [RCV1](http://www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf). The code used to preprocess the datasets can be found [here](data.py) and the performance of the models on these datasets can be found [here](results.csv).
