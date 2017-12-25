import re
import collections

import numpy as np
import sklearn.datasets


class TextDataset(object):

    def clean_text(self):
        """
        Tokenization & string cleaning.
        """
        # TODO: NLP preprocessing (use nltk?). Stemming, lemmatization etc.
        for i, string in enumerate(self.documents):
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"(\d+)", " NUM ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r"\(", " ( ", string)
            string = re.sub(r"\)", " ) ", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\?", " ? ", string)
            string = re.sub(r"\$", " dollar ", string)
            string = re.sub(r"\s{2,}", " ", string)
            self.documents[i] = string.strip().lower()

    def vectorize(self, **params):
        """
        Vectorize the documents in the dataset.
        """
        vectorizer = sklearn.feature_extraction.text.CountVectorizer(**params)
        self.data = vectorizer.fit_transform(self.documents)
        self.vocab = vectorizer.get_feature_names()

        assert len(self.vocab) == self.data.shape[1]

    def keep_documents(self, idx):
        """
        Keep the documents given by the index, discard the others.
        """
        self.documents = [self.documents[i] for i in idx]
        self.labels = self.labels[idx]
        try:
            self.data = self.data[idx, :]
        except AttributeError:
            pass

    def keep_words(self, idx):
        """
        Keep the words given by the index, discard the others.
        """
        self.data = self.data[:, idx]
        self.vocab = [self.vocab[i] for i in idx]

    def remove_short_documents(self, nwords, vocab="selected"):
        """
        Remove documents that contain less than nwords.
        """
        if vocab is "selected":
            # Word count with selected vocabulary.
            wc = self.data.sum(axis=1)
            wc = np.squeeze(np.asarray(wc))
        elif vocab is "full":
            # Word count with full vocabulary.
            wc = np.empty(len(self.documents), dtype=np.int)
            for i, doc in enumerate(self.documents):
                wc[i] = len(doc.split())
        idx = np.argwhere(wc >= nwords).squeeze()
        self.keep_documents(idx)

    def keep_top_words(self, N):
        """
        Keep in the vocaluary the N words that appear most often.
        """
        freq = self.data.sum(axis=0)
        freq = np.squeeze(np.asarray(freq))
        idx = np.argsort(freq)[::-1]
        idx = idx[:N]
        self.keep_words(idx)

    def normalize(self, norm="l1"):
        """
        TF-IDF transform & normalize data.
        """
        transformer = sklearn.feature_extraction.text.TfidfTransformer(norm=norm)
        self.data = transformer.fit_transform(self.data)


class Text20News(TextDataset):

    def __init__(self, subset, remove=("headers", "footers", "quotes"), categories=None,
                 shuffle=True, random_state=42):
        """
        http://scikit-learn.org/stable/datasets/twenty_newsgroups.html
        """
        dataset = sklearn.datasets.fetch_20newsgroups(subset=subset, categories=categories, shuffle=shuffle,
                                                      random_state=random_state, remove=remove)
        self.documents = dataset.data
        self.labels = dataset.target
        self.class_names = dataset.target_names
        assert max(self.labels) + 1 == len(self.class_names)

    def remove_encoded_images(self, freq=1e3):
        widx = self.vocab.index("ax")
        wc = self.data[:, widx].toarray().squeeze()
        idx = np.argwhere(wc < freq).squeeze()
        self.keep_documents(idx)


def one_hot_labels(num_labels, labels):
    """
    Generate one-hot encoded label arrays.
    """
    labels_arr = []
    for i in range(len(labels)):
        label = [0 for j in range(num_labels)]
        label[labels[i]] = 1
        labels_arr.append(label)
    y = np.array(labels_arr)

    return y


def load_word2vec(filepath, vocabulary, embedding_dim, tf_VP=False):
    """
    Returns the embedding matrix for vocabulary from filepath.
    """

    if tf_VP:
        null_index = 0
    else:
        null_index = None

    # Initialize embedding matrix from pre-trained word2vec embeddings. 0.25 is chosen so that unknown
    # vectors have (approximately) the same variance as pre-trained ones.
    embeddings = np.random.uniform(-0.25, 0.25, (len(vocabulary), embedding_dim))

    words_found = 0
    with open(filepath, "rb") as f:
        header = f.readline()
        word2vec_vocab_size, embedding_size = map(int, header.split())
        binary_len = np.dtype("float32").itemsize * embedding_size
        for line in range(word2vec_vocab_size):
            word = []
            while True:
                ch = f.read(1).decode("latin-1")
                if ch == " ":
                    word = "".join(word)
                    break
                if ch != "\n":
                    word.append(ch)

            if tf_VP:
                idx = vocabulary.get(word)
            else:
                idx = vocabulary.get(word, None)

            if idx != null_index:
                embeddings[idx] = np.fromstring(f.read(binary_len), dtype="float32")
                words_found += 1
            else:
                f.read(binary_len)

    print("Word Embeddings Extracted: {}".format(words_found))
    print("Word Embeddings Randomly Initialized: {}".format(len(vocabulary) - words_found))

    return embeddings


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    indices = collections.deque()
    num_iterations = int(num_epochs * data_size / batch_size)
    for step in range(1, num_iterations + 1):
        if len(indices) < batch_size:
            if shuffle:
                indices.extend(np.random.permutation(data_size))
            else:
                indices.extend(np.arange(data_size))
        idx = [indices.popleft() for i in range(batch_size)]
        yield data[idx]
