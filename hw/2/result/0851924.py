import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gensim
import gc
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
OUTPUT_FILE = "ans_test.csv"

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
        new_doc = ['</s>']*(seq_len - len(doc))
        new_doc += doc
        return new_doc
    else:
        return doc


def document_vector(vocab, doc):
    doc = [word for word in doc if word in vocab]
    doc = pad_zeros(doc)

    return [vocab[word].index for word in doc]


if __name__ == '__main__':

    # move model to GPU, if available
    train_on_gpu=torch.cuda.is_available()
    if(train_on_gpu):
        print('Training on GPU.')

    # load Glove pre train model
    # _ = glove2word2vec(MODEL_DIR+MODEL_FILE, MODEL_DIR+VEC_MODEL_FILE)
    model = KeyedVectors.load_word2vec_format(MODEL_DIR+VEC_MODEL_FILE, binary=False)
    # check dimension of word vectors
    print(model.vectors.shape)

    weights = torch.FloatTensor(model.vectors)
    vocab_dict = model.vocab.copy()
    print(weights.shape)

    # remove model to free RAM
    # %xdel model
    # %reset out
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

    full_dataset = TextDataset(data_only_df)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # test the loader
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()
    print('Sample input size: ', sample_x.size())  # batch_size, seq_length
    print('Sample input: \n', sample_x)
    print()
    print('Sample label size: ', sample_y.size())  # batch_size
    print('Sample label: \n', sample_y)

    net = AttractivenessRNN(weights, weights.shape[1], 32, 2)
    if(train_on_gpu):
        net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    criterion = nn.MSELoss()
    counter = 0
    print_every = 100
    clip = 5  # gradient clipping

    net.train()
    # train for some number of epochs
    for e in range(EPOCH):
        # initialize hidden state
        h = net.init_hidden(BATCH_SIZE, train_on_gpu)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            if(train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            # nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # get validation loss
                val_h = net.init_hidden(BATCH_SIZE, train_on_gpu)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:

                    # creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    if(train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())
                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e+1, EPOCH),
                    "Step: {}...".format(counter),
                    "Loss: {:.6f}...".format(loss.item()),
                    "Val Loss: {:.6f}".format(np.mean(val_losses)))
    net.eval()

    sample_input = test_data_only_df.iloc[0]['word_list']
    sample_input = torch.LongTensor([sample_input])
    sample_input.size()

    h = net.init_hidden(1, train_on_gpu)
    h = tuple([each.data for each in h])
    pred, h = net(sample_input.cuda(), h)
    pred.squeeze().item()

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
