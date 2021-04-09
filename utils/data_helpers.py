import os, json
import numpy as np
from tflearn.data_utils import pad_sequences
from scipy.sparse import lil_matrix


def get_onehot_label_threshold(scores, threshold=0.5):
    """
    Get the predicted onehot labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.

    Args:
        scores: The all classes predicted scores provided by network
        threshold: The threshold (default: 0.5)
    Returns:
        predicted_onehot_labels: The predicted labels (onehot)
    """
    scores = np.array(scores)
    predicted_onehot_labels = np.zeros(scores.shape)
    predicted_onehot_labels[np.array(scores) >= threshold] = 1
    scores_max = np.argmax(scores, axis=-1)
    predicted_onehot_labels[np.array(list(range(len(scores)))), scores_max] = 1
    return predicted_onehot_labels


def get_onehot_label_topk(scores, top_num=1):

    scores = np.array(scores)
    predicted_onehot_labels = np.zeros(scores.shape)
    y_index = np.argsort(scores, axis=-1)[:, -top_num:]
    x_index = np.reshape(np.array(list(range(len(scores)))), newshape=[-1, 1])
    predicted_onehot_labels[x_index, y_index] = 1
    return predicted_onehot_labels


def get_label_threshold(scores, threshold=0.5):

    scores = np.array(scores)
    predicted_onehot_labels = np.zeros(scores.shape)
    predicted_onehot_labels[np.array(scores) >= threshold] = 1
    scores_max = np.argmax(scores, axis=-1)
    predicted_onehot_labels[np.array(list(range(len(scores)))), scores_max] = 1
    return lil_matrix(predicted_onehot_labels)


def get_label_topk(scores, top_num=1):

    scores = np.array(scores)
    predicted_onehot_labels = np.zeros(scores.shape)
    y_index = np.argsort(scores, axis=-1)[:, -top_num:]
    x_index = np.reshape(np.array(list(range(len(scores)))), newshape=[-1, 1])
    predicted_onehot_labels[x_index, y_index] = 1
    return lil_matrix(predicted_onehot_labels)

def load_glove_word_embedding(embedding_size, glove_file):
    if not os.path.isfile(glove_file):
        print("✘ The GloVe file {} doesn't exist. ".format(glove_file))
        return 0, None, dict()
    vocab_size = len(open(glove_file,'r').readlines())+1
    vector_matrix = np.zeros([vocab_size, embedding_size])
    word2id = dict()
    word_id = 1
    with open(glove_file,'r') as f_in:
        for line in f_in:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            word2id[word] = word_id
            vector_matrix[word_id] = vector
            word_id += 1

    return vocab_size, vector_matrix, word2id



def load_data_and_labels(data_file, num_classes_list, embedding_size, word2id_dict, increase_word2id = False):
    """
    Args:
        data_file: The research data
        num_classes_list: <list> The number of classes
        embedding_size: The embedding size
        data_aug_flag: The flag of data augmented
    Returns:
        The class Data
    """
    def _token_to_index(content):
        result = []
        if increase_word2id:
            for item in content:
                if item not in word2id_dict:
                    word2id_dict[item] = len(word2id_dict)
                result.append(word2id_dict[item])
        else:
            for item in content:
                word2id = word2id_dict.get(item)
                if word2id is None:
                    word2id = 0
                result.append(word2id)
        return result

    if not data_file.endswith('.json'):
        raise IOError("✘ The research data is not a json file. "
                      "Please preprocess the research data into the json file.")
    if word2id_dict is None:
        new_word2id_dict = dict()
    with open(data_file) as fin:
        id_list = []
        content_index_list = []
        labels_list = []
        labels_tuple_list = []
        total_line = 0

        for eachline in fin:
            data = json.loads(eachline)
            patent_id = data['id']

            id_list.append(patent_id)
            content_index_list.append(_token_to_index(data['content']))
            labels_list.append(data['label_combine'])
            labels_tuple_list.append(tuple(i for i in data['label_local']))
            total_line += 1

    class _Data:
        def __init__(self):
            pass

        @property
        def number(self):
            return total_line

        @property
        def patent_id(self):
            return id_list

        @property
        def content_tokenindex(self):
            return content_index_list

        @property
        def labels(self):
            return labels_list

        @property
        def labels_tuple(self):
            return labels_tuple_list

    return _Data(), word2id_dict

def batch_iter(data, batch_size, num_epochs, pad_seq_len, num_classes_list, total_classes, shuffle=True):
    def _create_onehot_labels(labels_index, num_labels):
        label = [0] * num_labels
        for item in labels_index:
            label[int(item)] = 1
        return label
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            x_batch_train, y_batch_train, y_batch_train_tuple = zip(*shuffled_data[start_index:end_index])
            x_batch_train = pad_sequences(x_batch_train, maxlen=pad_seq_len, value=0.)
            y_batch_train_onehot = []
            y_batch_train_tuple_onehot = [[] for i in num_classes_list]
            for i in range(len(y_batch_train)):
                y_batch_train_onehot.append(_create_onehot_labels(y_batch_train[i], total_classes))
                for idx, num_class in enumerate(num_classes_list):
                    y_batch_train_tuple_onehot[idx].append(
                        _create_onehot_labels(y_batch_train_tuple[i][idx], num_class))
            yield x_batch_train,y_batch_train_onehot, y_batch_train_tuple_onehot

def batch_iter_test(data, batch_size, num_epochs, pad_seq_len, num_classes_list, total_classes, shuffle=True):
    def _create_onehot_labels(labels_index, num_labels):
        label = [0] * num_labels
        for item in labels_index:
            label[int(item)] = 1
        return label
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            x_batch_test, y_batch_test, y_batch_test_tuple = zip(*shuffled_data[start_index:end_index])
            x_batch_test = pad_sequences(x_batch_test, maxlen=pad_seq_len, value=0.)
            y_batch_test_onehot = []
            y_batch_test_tuple_onehot = [[] for i in num_classes_list]
            for i in range(len(y_batch_test)):
                y_batch_test_onehot.append(_create_onehot_labels(y_batch_test[i], total_classes))
                for idx, num_class in enumerate(num_classes_list):
                    y_batch_test_tuple_onehot[idx].append(
                        _create_onehot_labels(y_batch_test_tuple[i][idx], num_class))
            yield x_batch_test,y_batch_test_onehot, y_batch_test_tuple_onehot

class logger():
    def __init__(self, log_file):
        self.f_out = open(log_file,'w')

    def print(self, string):
        print(string)
        self.f_out.write(string+'\n')

    def close(self):
        self.f_out.close()