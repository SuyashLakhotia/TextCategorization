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
            string = re.sub(r"[^A-Za-z0-9(),!?'$]", " ", string)
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

    def keep_documents(self, idx):
        """
        Keep the documents given by the index, discard the others.
        """
        self.documents = [self.documents[i] for i in idx]
        self.labels = self.labels[idx]
        try:
            self.data_count = self.data_count[idx, :]
        except AttributeError:
            pass

    def keep_words(self, idx):
        """
        Keep the words given by the index, discard the others.
        """
        self.vocab = [self.vocab[i] for i in idx]
        self.data_count = self.data_count[:, idx]

    def remove_short_documents(self, nwords, vocab="selected"):
        """
        Remove documents that contain less than nwords.
        """
        if vocab is "selected":
            # Word count with selected vocabulary.
            wc = self.data_count.sum(axis=1)
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
        Keep only the N words that appear most often.
        """
        freq = self.data_count.sum(axis=0)
        freq = np.squeeze(np.asarray(freq))
        idx = np.argsort(freq)[::-1]
        idx = idx[:N]
        self.keep_words(idx)

    def count_vectorize(self, **params):
        """
        Vectorize the documents in the dataset using CountVectorizer(**params).
        """
        self.count_vectorizer = sklearn.feature_extraction.text.CountVectorizer(**params)
        self.data_count = self.count_vectorizer.fit_transform(self.documents)
        self.vocab = self.count_vectorizer.get_feature_names()
        assert len(self.vocab) == self.data_count.shape[1]

    def tfidf_normalize(self, norm="l1"):
        """
        TF-IDF transform & normalize data_count. Do this at the very end.
        """
        transformer = sklearn.feature_extraction.text.TfidfTransformer(norm=norm)
        self.data_tfidf = transformer.fit_transform(self.data_count)

    def generate_word2ind(self, maxlen=None, padding="post", truncating="post"):
        """
        Transforms documents to list of self.vocab indexes of the same length (i.e. maxlen). Do this at the
        very end.
        """
        # Add "<UNK>" to vocabulary and create a reverse vocabulary lookup.
        if self.vocab[-1] != "<UNK>":
            self.vocab = self.vocab + ["<UNK>"]
        reverse_vocab = {w: i for i, w in enumerate(self.vocab)}

        # Tokenize all the documents using the CountVectorizer's analyzer.
        analyzer = self.count_vectorizer.build_analyzer()
        tokenized_docs = np.array([analyzer(doc) for doc in self.documents])

        # Transform documents from words to indexes using svocabulary.
        sequences = np.array([[reverse_vocab[w] for w in tokens if w in reverse_vocab]
                              for tokens in tokenized_docs])

        # Truncate or pad sequences to match maxlen. Adapted from tflearn.data_utils.pad_sequences.
        lengths = [len(s) for s in sequences]
        num_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)

        x = np.ones((num_samples, maxlen), np.int64) * (len(self.vocab) - 1)
        for idx, s in enumerate(sequences):
            if len(s) == 0:
                continue  # empty list was found
            if truncating == "pre":
                trunc = s[-maxlen:]
            elif truncating == "post":
                trunc = s[:maxlen]

            if padding == "post":
                x[idx, :len(trunc)] = trunc
            elif padding == "pre":
                x[idx, -len(trunc):] = trunc

        self.data_word2ind = x


class Text20News(TextDataset):
    """
    20 Newsgroups Dataset
    http://scikit-learn.org/stable/datasets/twenty_newsgroups.html
    """

    def __init__(self, subset, remove=("headers", "footers", "quotes"), categories=None,
                 shuffle=True, random_state=42):
        dataset = sklearn.datasets.fetch_20newsgroups(subset=subset, categories=categories, shuffle=shuffle,
                                                      random_state=random_state, remove=remove)
        self.documents = dataset.data
        self.labels = dataset.target
        self.class_names = dataset.target_names
        assert max(self.labels) + 1 == len(self.class_names)

    def remove_encoded_images(self, freq=1e3):
        widx = self.vocab.index("ax")
        wc = self.data_count[:, widx].toarray().squeeze()
        idx = np.argwhere(wc < freq).squeeze()
        self.keep_documents(idx)

    def preprocess_train(self, out, **params):
        self.remove_short_documents(nwords=20, vocab="full")  # remove documents < 20 words in length
        self.clean_text()  # tokenize & clean text
        self.count_vectorize(stop_words="english")  # create term-document count matrix and vocabulary
        self.orig_vocab_size = len(self.vocab)
        self.remove_encoded_images()  # remove encoded images
        self.keep_top_words(10000)  # keep only the top 10000 words
        self.remove_short_documents(nwords=5, vocab="selected")  # remove docs whose signal would be the zero vector

        if out == "tfidf":
            self.tfidf_normalize(**params)  # transform count matrix into a normalized tf-idf matrix
        elif out == "word2ind":
            self.generate_word2ind(**params)  # transform documents to sequences of vocab indexes

    def preprocess_test(self, train_vocab, out, **params):
        self.clean_text()
        self.count_vectorize(vocabulary=train_vocab)
        self.remove_short_documents(nwords=5, vocab="selected")

        if out == "tfidf":
            self.tfidf_normalize(**params)
        elif out == "word2ind":
            self.generate_word2ind(**params)


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


def load_word2vec(filepath, vocabulary, embedding_dim):
    """
    Returns the embedding matrix for vocabulary from filepath.
    """
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

            idx = vocabulary.get(word, None)
            if idx != None:
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
