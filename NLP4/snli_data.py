# Avital Rose 318413408
import argparse

import torch
from torch.utils.data import Dataset
import pickle
import numpy as np


def sort_by_length_keep_indexes(to_sort):
    """
    Function to sort by premise length
    :param to_sort:
    :return:
    """
    ordered_sentences = sorted(to_sort, key=lambda x: x[1], reverse=True)
    return ordered_sentences


class SnliData:
    def __init__(self, args):
        self.args = args
        self.word_to_index = {}
        self.index_to_word = {}
        self.labels_to_index = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        self.pad = '<PAD>'
        self.null_word = '<NULL>'
        self.make_dictionary()
        self.embedding_size = args.embedding_dim
        self.word_embedding, self.index_embedding = self.get_word_embedding()
        self.train, self.dev, self.test = self.get_indexed_data()

    def make_dictionary(self):
        """
        Function to make word to index and index to word dictionaries
        :return:
        """
        with open(self.args.file_train, "r") as f:
            lines = f.readlines()

        words_list = []
        for idx, line in enumerate(lines):
            if idx == 0:
                continue  # line 0 has different format

            # read premise and hypothesis.
            # place 5 is sentence1 and place 6 is sentence2 according to first line description
            premise = [w.lower() for w in line.split('\t')[1].split(' ') if w != "(" and w != ")"]
            hypothesis = [w.lower() for w in line.split('\t')[2].split(' ') if w != "(" and w != ")"]
            premise_and_hypothesis = premise + hypothesis

            # make dictionary
            for word in premise_and_hypothesis:
                words_list.append(word)

        self.word_to_index = {w: i for i, w in enumerate(list(set(words_list)), start=2)}
        self.index_to_word = {i: w for w, i in self.word_to_index.items()}
        self.word_to_index[self.pad] = 0
        self.index_to_word[0] = self.pad
        self.word_to_index[self.null_word] = 1
        self.index_to_word[1] = self.null_word

    def get_word_embedding(self):
        """
        Function returns word to vector embedding as well as index to vector embedding.
        :return:
        """
        word_to_vector = {}
        with open("glove.840B.300d.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            word = line.split(' ')[0]
            vector = line.split('\n')[0].split(' ')[1:]
            if word in self.word_to_index.keys():
                word_to_vector[word] = [float(v) for v in vector]

        # adding padding and null vectors
        word_to_vector[self.pad] = [0.0] * self.embedding_size
        word_to_vector[self.null_word] = [1.0] * self.embedding_size

        index_to_vector = np.zeros((len(self.index_to_word.keys()), self.embedding_size))
        for index in self.index_to_word.keys():
            index_to_vector[index] = word_to_vector.get(self.index_to_word[index], np.ones(self.embedding_size))

        # development
        index_to_vector[self.word_to_index[self.pad]] = [0.0] * self.embedding_size
        index_to_vector[self.word_to_index[self.null_word]] = [1.0] * self.embedding_size
        #
        # # used while developing for saving variables in between runs
        # with open('word_to_vector.pkl', 'wb') as file:
        #     # A new file will be created
        #     pickle.dump(word_to_vector, file)
        # with open('index_to_vector.pkl', 'wb') as file:
        #     # A new file will be created
        #     pickle.dump(index_to_vector, file)
        # also used for development
        # with open('word_to_vector.pkl', 'rb') as file:
        #     # Call load method to deserialize
        #     word_to_vector = pickle.load(file)
        # with open('index_to_vector.pkl', 'rb') as file:
        #     # Call load method to deserialize
        #     index_to_vector = pickle.load(file)
        return word_to_vector, index_to_vector

    def get_indexed_data(self):
        """
        Function to return loaded data for train, dev and test
        :return:
        """
        train_data = self.load_data(self.args.file_train)
        dev_data = self.load_data(self.args.file_dev)
        test_data = self.load_data(self.args.file_test)
        return train_data, dev_data, test_data

    def load_data(self, file_name):
        """
        Function to load data for given file.
        The data is made up of the list of indexes for all the words in sentence (premise and hypothesis) as well as the
        premise and hypothesis sentence's lengths and label for the duo.
        :param file_name: file name to load data from
        :return: list of lists [premise_indexes, premise_len, hypothesis_indexes, hypothesis_len, y]
        """
        data = []
        pad_index = self.word_to_index[self.pad]
        null_index = self.word_to_index[self.null_word]

        with open(file_name, "r") as f:
            lines = f.readlines()

        # iterate over all sentences in file
        for index, line in enumerate(lines):
            if index == 0:  # first line is information, not sentences
                continue
            # read premise and hypothesis.
            # place 5 is sentence1 and place 6 is sentence2 according to first line description
            premise = [w.lower() for w in line.split('\t')[1].split(' ') if w != "(" and w != ")"]
            hypothesis = [w.lower() for w in line.split('\t')[2].split(' ') if w != "(" and w != ")"]
            y = int(self.labels_to_index.get(line.split(' ')[0].split('\t')[0], 0))

            # find lengths for future padding if needed
            premise_len = len(premise)
            hypothesis_len = len(hypothesis)

            # convert to word indexes, if not in word_to_index
            premise_indexes = [self.word_to_index.get(w, null_index) for w in premise]
            hypothesis_indexes = [self.word_to_index.get(w, null_index) for w in hypothesis]

            # make all the same length, possibly will be removed after development
            # fix premise lengths
            if len(premise_indexes) > self.args.max_premise:
                self.args.max_premise = len(premise_indexes)
            while len(premise_indexes) < self.args.max_premise:
                premise_indexes.append(pad_index)
            assert len(premise_indexes) == self.args.max_premise

            # fix hypothesis lengths
            if len(hypothesis_indexes) > self.args.max_hypothesis:
                self.args.max_hypothesis = len(hypothesis_indexes)
            while len(hypothesis_indexes) < self.args.max_hypothesis:
                hypothesis_indexes.append(pad_index)
            assert len(hypothesis_indexes) == self.args.max_hypothesis

            data.append([premise_indexes, premise_len, hypothesis_indexes, hypothesis_len, y])
        data = sort_by_length_keep_indexes(to_sort=data)
        return data

    def get_data_loaders(self):
        """
        Function to return data loaders for train, dev and test
        :return:
        """
        train_dataset = SnliDataset(self.train)
        dev_dataset = SnliDataset(self.dev)
        test_dataset = SnliDataset(self.test)
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                                   batch_size=self.args.batch_size, collate_fn=self.collate_function)
        dev_loader = torch.utils.data.DataLoader(dev_dataset, shuffle=self.args.shuffle,
                                                 batch_size=self.args.batch_size, collate_fn=self.collate_function)
        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=self.args.shuffle,
                                                  batch_size=self.args.batch_size, collate_fn=self.collate_function)
        return train_loader, dev_loader, test_loader

    def collate_function(self, inputs):
        """
        Function to reorganize data for batching.
        The function gets a list where each item is made up of premise and hypothesis dou i.e. list of premise indexes,
        premise length, hypothesis indexes, hypothesis length, and label for relationship between premise and hypothesis.
        The function reorganizes the inputs to be list of premise indexes, list of premise lengths, list of hypothesis
        indexes, list of hypothesis lengths and finally list of labels.
        :param inputs: list of lists
        :return:
        """
        # get data in proper format
        premises = [x[0] for x in inputs]
        hypothesises = [x[2] for x in inputs]
        lengths_premise = [x[1] for x in inputs]
        lengths_hypothesis = [x[3] for x in inputs]
        y = [x[4] for x in inputs]

        # make tensors
        premise_tensor = torch.from_numpy(np.array(premises))
        hypothesises_tensor = torch.from_numpy(np.array(hypothesises))
        y = torch.from_numpy(np.array(y))

        return premise_tensor, lengths_premise, hypothesises_tensor, lengths_hypothesis, y


class SnliDataset(Dataset):
    """
    Class to Make custom dataset for SNLI data
    """
    def __init__(self, inputs):
        self.inputs = inputs

    def __getitem__(self, item):
        """
        Function to return singular item from data
        :param item: index of data to return
        :return:
        """
        return self.inputs[item]

    def __len__(self):
        """
        Function returns length of data
        :return:
        """
        return len(self.inputs)


if __name__ == "__main__":
    # used while developing
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-train', default="snli_1.0/snli_1.0_train.txt")
    parser.add_argument('--file-dev', default="snli_1.0/snli_1.0_dev.txt")
    parser.add_argument('--file-test', default="snli_1.0/snli_1.0_test.txt")
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--embedding-dim', default=300, type=int)
    parser.add_argument('--max-premise', default=83, type=int)
    parser.add_argument('--max-hypothesis', default=83, type=int)
    parser.add_argument('--shuffle', default=False)
    parser.add_argument('--num-heads', default=5, type=int)
    parser.add_argument('--alpha', default=1.5, type=float)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    args = parser.parse_args()
    _data = SnliData(args)

