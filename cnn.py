import os
import numpy as np
import gensim
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Embedding, Input, Dense, Dropout, 
    Conv1D, GlobalMaxPooling1D, Concatenate 
)
from tensorflow.keras import utils as np_utils
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import argparse

# Import các file hỗ trợ từ project của bạn
from data_handler import get_data
from my_tokenizer import glove_tokenize

# Cấu hình cho bài toán Privacy
NO_OF_CLASSES = 2 
SEED = 42
NO_OF_FOLDS = 10
EPOCHS = 10
BATCH_SIZE = 64

vocab = {}

def gen_vocab(tweets, tokenizer):
    vocab_index = 1
    for tweet in tweets:
        words = tokenizer(tweet['text'].lower())
        for word in words:
            if word not in vocab:
                vocab[word] = vocab_index
                vocab_index += 1
    vocab['UNK'] = len(vocab) + 1
    return vocab

def gen_sequence(tweets, tokenizer, vocab):
    X, y = [], []
    for tweet in tweets:
        words = tokenizer(tweet['text'].lower())
        seq = [vocab.get(word, vocab['UNK']) for word in words]
        X.append(seq)
        y.append(tweet['label'])
    return X, y

def get_embedding_weights(word2vec_model, vocab, embedding_dim):
    embedding = np.zeros((len(vocab) + 2, embedding_dim))
    for k, v in vocab.items():
        try:
            embedding[v] = word2vec_model[k]
        except:
            pass
    return embedding

def cnn_model(sequence_length, embedding_dim, vocab_size, learn_embeddings):
    filter_sizes = (2, 3, 4)
    num_filters = 100
    graph_in = Input(shape=(sequence_length,))
    emb = Embedding(vocab_size, embedding_dim, trainable=learn_embeddings)(graph_in)
    emb = Dropout(0.25)(emb)

    convs = []
    for fsz in filter_sizes:
        conv = Conv1D(filters=num_filters, kernel_size=fsz, padding='same', activation='relu')(emb)
        pool = GlobalMaxPooling1D()(conv)
        convs.append(pool)

    out = Concatenate()(convs) if len(convs) > 1 else convs[0]
    out = Dropout(0.5)(out)
    out = Dense(NO_OF_CLASSES, activation='softmax')(out)

    model = Model(inputs=graph_in, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Cập nhật đúng tên file glove.twitter.27B.25d.txt và dimension 25
    parser.add_argument('-f', '--embeddingfile', default='./tweet_data/glove.twitter.27B.25d.txt')
    parser.add_argument('-d', '--dimension', type=int, default=25)
    parser.add_argument('--learn-embeddings', action='store_true', default=True)
    args = parser.parse_args()

    print(f'Đang tải file GloVe: {args.embeddingfile}...')
    # File GloVe Twitter này thường không có dòng header
    try:
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
            args.embeddingfile, binary=False, no_header=True
        )
    except Exception as e:
        print(f"Lưu ý: Đang thử cách tải thứ hai do: {e}")
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(args.embeddingfile)
    
    raw_tweets = get_data() 
    vocab = gen_vocab(raw_tweets, glove_tokenize)
    X, y = gen_sequence(raw_tweets, glove_tokenize, vocab)
    
    MAX_SEQUENCE_LENGTH = max(len(x) for x in X)
    data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y_cat = np_utils.to_categorical(y, num_classes=NO_OF_CLASSES)
    
    W = get_embedding_weights(word2vec_model, vocab, args.dimension)
    
    cv = KFold(n_splits=NO_OF_FOLDS, shuffle=True, random_state=SEED)
    for i, (train, test) in enumerate(cv.split(data)):
        print(f"\n--- Fold {i+1} ---")
        model = cnn_model(MAX_SEQUENCE_LENGTH, args.dimension, len(vocab)+2, args.learn_embeddings)
        model.layers[1].set_weights([W]) 
        
        model.fit(data[train], y_cat[train], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        y_pred = np.argmax(model.predict(data[test]), axis=1)
        print(classification_report(np.argmax(y_cat[test], axis=1), y_pred))