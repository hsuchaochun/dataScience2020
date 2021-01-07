import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gensim
import gc
import gensim.models.keyedvectors as w2v
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from hw2_lib.attractiveness import TextDataset
from hw2_lib.attractiveness import AttractivenessRNN

# training params
LR = 0.005
EPOCH = 4  # 3-4 is approx where I noticed the validation loss stop decreasing
BATCH_SIZE = 32

# other parameters
DATASET_DIR = "./dataset/"
TRAINING_DATASET_FILE = "train.csv"
TESTING_DATASET_FILE = "test.csv"
MODEL_DIR = "./models/"
MODEL_FILE = "glove.840B.300d.txt"
VEC_MODEL_FILE = "glove_vec.840B.300d.txt"
OUTPUT_DIR = "./outputs/"
OUTPUT_FILE = "ans_final.csv"

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

seq_len = 0

def preprocess(text):
    # word to lowercase
    text = text.lower()
    doc = word_tokenize(text)
    # remove stop words
    doc = [word for word in doc if word not in stop_words]
    # remove strings contain non-alphabets
    doc = [word for word in doc if word.isalpha()]

    return doc


def pad_zeros(doc):
    if len(doc) < seq_len:
        new_doc = ['<']*(seq_len - len(doc))
        new_doc += doc
        return new_doc
    else:
        return doc


def document_vector(vocab, doc):
    doc = [word for word in doc if word in vocab]
    doc = pad_zeros(doc)

    return [vocab[word].index for word in doc]

def restrict_w2v(w2v, restricted_word_set):
    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    new_vectors_norm = []

    for i in range(len(w2v.vocab)):
        word = w2v.index2entity[i]
        vec = w2v.vectors[i]
        vocab = w2v.vocab[word]
        vec_norm = w2v.vectors_norm[i]
        if word in restricted_word_set:
            vocab.index = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            new_vectors.append(vec)
            new_vectors_norm.append(vec_norm)
    w2v.vocab = new_vocab
    w2v.vectors = np.array(new_vectors)
    w2v.index2entity = np.array(new_index2entity)
    w2v.index2word = np.array(new_index2entity)


if __name__ == '__main__':

    # move model to GPU, if available
    train_on_gpu = torch.cuda.is_available()
    if(train_on_gpu):
        print('Training on GPU.')

    # load Glove pre train model
    # _ = glove2word2vec(MODEL_DIR+MODEL_FILE, MODEL_DIR+VEC_MODEL_FILE)
    # model = KeyedVectors.load_word2vec_format(
        # MODEL_DIR+VEC_MODEL_FILE, binary=False)
    # check dimension of word vectors
    model = gensim.models.KeyedVectors.load_word2vec_format('./models/word2vec.model.bin', binary=False)
    print(model.vectors.shape)

    weights = torch.FloatTensor(model.vectors)
    vocab_dict = model.vocab.copy()

    # remove model to free RAM
    # del model
    # reset out
    gc.collect()

    main_data = pd.read_csv(DATASET_DIR+TRAINING_DATASET_FILE)
    main_data = main_data.drop(columns=['ID'])
    test_data = pd.read_csv(DATASET_DIR+TESTING_DATASET_FILE)
    test_data = test_data.drop(columns=['ID'])
    main_data = main_data.append(test_data, ignore_index=True)
    # main_data.head()

    headlines = main_data['Headline']
    # headlines.shape

    headlines_list = [headline for headline in headlines]
    corpus = [preprocess(headline) for headline in headlines_list]
    print(type(corpus[0]))

    # pre-padding the headlines to get fixed-size input
    length_of_sentences = [len(doc) for doc in corpus]
    seq_len = max(length_of_sentences)

    x = []
    # append the vector for each document
    for doc in corpus:
        x.append(document_vector(vocab_dict, doc))

    # concatenate list into dataframe
    index_df = pd.DataFrame()
    index_df['word_list'] = x

    main_w_vectors = pd.concat((index_df, main_data), axis=1)
    main_w_vectors.head()

    # drop all non-numeric, non-dummy columns
    cols_to_drop = ['Headline', 'Category']
    data_only_df = main_w_vectors.drop(columns=cols_to_drop)

    # remove test data from training data
    test_data_only_df = data_only_df[-227:]
    data_only_df = data_only_df[:-227]

    net = AttractivenessRNN(weights, weights.shape[1], 8, 2)
    net.load_state_dict(torch.load('./weights/weights.pth'))
    if (train_on_gpu):
        net.cuda()
    else:
        net.cpu()
    
    net.eval()

    pred_input = test_data_only_df['word_list'].values.tolist()
    pred_input = torch.LongTensor(pred_input)
    pred_input.size()

    h = net.init_hidden(227, train_on_gpu)
    h = tuple([each.data for each in h])
    pred, h = net(pred_input.cuda(), h)

    pred_output = pred.data.tolist()

    # output prediction
    new_df = pd.DataFrame()
    new_df['Label'] = pred_output
    new_df.index = np.arange(1, len(new_df) + 1)
    new_df.to_csv(OUTPUT_DIR + OUTPUT_FILE, index_label='ID')
