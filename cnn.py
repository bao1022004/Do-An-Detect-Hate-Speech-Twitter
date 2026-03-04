import os
import sys
import codecs
import operator
import numpy as np
import pdb
import gensim
import sklearn
from collections import defaultdict
from string import punctuation

# Kích hoạt chế độ tương thích v1 cho TensorFlow 2.x
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
tf.config.run_functions_eagerly(True)

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, Input, LSTM, Activation, Dense, Dropout, 
    Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    Concatenate 
)
from tensorflow.keras import utils as np_utils

from sklearn.metrics import (
    f1_score, precision_score, recall_score, classification_report
)
from sklearn.model_selection import KFold
from gensim.parsing.preprocessing import STOPWORDS

from data_handler import get_data
from batch_gen import batch_gen
from my_tokenizer import glove_tokenize
from nltk import tokenize as tokenize_nltk
import argparse

vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = []

EMBEDDING_DIM = None
GLOVE_MODEL_FILE = None
NO_OF_CLASSES = 7 

SEED = 42
NO_OF_FOLDS = 10
LOSS_FUN = None
OPTIMIZER = None
TOKENIZER = None
INITIALIZE_WEIGHTS_WITH = None
LEARN_EMBEDDINGS = None
EPOCHS = 10
BATCH_SIZE = 128

word2vec_model = None

def get_embedding_weights():
    embedding = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
    n = 0
    for k, v in vocab.items():
        try:
            embedding[v] = word2vec_model[k]
        except:
            n += 1
    print("%d embedding missed" % n)
    return embedding

def select_tweets():
    raw_tweets = get_data()
    tweet_return = []
    for tweet in raw_tweets:
        words = TOKENIZER(tweet['text'].lower())
        _emb = sum(1 for w in words if w in word2vec_model)
        if _emb > 0:
            tweet_return.append(tweet)
    print('Tweets selected:', len(tweet_return))
    return tweet_return

def gen_vocab():
    vocab_index = 1
    for tweet in tweets:
        words = TOKENIZER(tweet['text'].lower())
        # Loại bỏ dấu câu và stopword đúng cách
        clean_words = [w.strip(punctuation) for w in words]
        clean_words = [w for w in clean_words if w and w not in STOPWORDS]

        for word in clean_words:
            if word not in vocab:
                vocab[word] = vocab_index
                reverse_vocab[vocab_index] = word
                vocab_index += 1
            freq[word] += 1
    vocab['UNK'] = len(vocab) + 1

def gen_sequence():
    X, y = [], []
    for tweet in tweets:
        words = TOKENIZER(tweet['text'].lower())
        clean_words = [w.strip(punctuation) for w in words]
        clean_words = [w for w in clean_words if w and w not in STOPWORDS]
        
        seq = [vocab.get(word, vocab['UNK']) for word in clean_words]
        X.append(seq)
        y.append(tweet['label'])
    return X, y

def cnn_model(sequence_length, embedding_dim):
    n_classes = NO_OF_CLASSES
    filter_sizes = (2, 3, 4) # Giảm kích thước filter để an toàn
    num_filters = 100
    dropout_prob = (0.25, 0.5)

    graph_in = Input(shape=(sequence_length,))
    emb = Embedding(len(vocab) + 1, embedding_dim, trainable=LEARN_EMBEDDINGS)(graph_in)
    emb = Dropout(dropout_prob[0])(emb)

    convs = []
    for fsz in filter_sizes:
        conv = Conv1D(filters=num_filters, kernel_size=fsz, padding='same', activation='relu')(emb)
        pool = GlobalMaxPooling1D()(conv)
        convs.append(pool)

    out = Concatenate()(convs) if len(filter_sizes) > 1 else convs[0]
    out = Dropout(dropout_prob[1])(out)
    out = Dense(n_classes, activation='softmax')(out)

    model = Model(inputs=graph_in, outputs=out)
    model.compile(loss=LOSS_FUN, optimizer=OPTIMIZER, metrics=['accuracy'])
    return model

def train_CNN(X, y, model, weights):
    cv_object = KFold(n_splits=NO_OF_FOLDS, shuffle=True, random_state=SEED)
    for train_index, test_index in cv_object.split(X):
        if INITIALIZE_WEIGHTS_WITH == "glove":
            model.layers[1].set_weights([weights])
        
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        y_train_cat = np_utils.to_categorical(y_train, num_classes=NO_OF_CLASSES)
        
        model.fit(X_train, y_train_cat, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--embeddingfile', required=True)
    parser.add_argument('-d', '--dimension', required=True)
    parser.add_argument('--tokenizer', choices=['glove', 'nltk'], required=True)
    parser.add_argument('--loss', required=True)
    parser.add_argument('--optimizer', required=True)
    parser.add_argument('--epochs', default=10)
    parser.add_argument('--batch-size', default=128)
    parser.add_argument('--initialize-weights', choices=['random', 'glove'], required=True)
    parser.add_argument('--learn-embeddings', action='store_true')
    args = parser.parse_args()

    EMBEDDING_DIM = int(args.dimension)
    GLOVE_MODEL_FILE = args.embeddingfile
    LOSS_FUN = args.loss
    OPTIMIZER = args.optimizer
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)
    INITIALIZE_WEIGHTS_WITH = args.initialize_weights
    LEARN_EMBEDDINGS = args.learn_embeddings

    if args.tokenizer == "nltk":
        TOKENIZER = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize
    else:
        TOKENIZER = glove_tokenize

    print('Loading Glove Model...')
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(GLOVE_MODEL_FILE)
    
    tweets = select_tweets()
    gen_vocab()
    X, y = gen_sequence()
    
    MAX_SEQUENCE_LENGTH = max(len(x) for x in X)
    print("Max sequence length: %d" % MAX_SEQUENCE_LENGTH)
    
    data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y = np.array(y)
    W = get_embedding_weights()
    model = cnn_model(data.shape[1], EMBEDDING_DIM)
    train_CNN(data, y, model, W)