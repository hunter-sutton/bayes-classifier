# This file implements a Naive Bayes Classifier
import numpy as np

class BayesClassifier():
    """
    Naive Bayes Classifier
    file length: file length of training file
    sections: sections for incremental training
    """
    def __init__(self):
        self.positive_word_counts = {}
        self.negative_word_counts = {}
        self.percent_positive_sentences = 0
        self.percent_negative_sentences = 0
        self.file_length = 499
        self.file_sections = [self.file_length // 4, self.file_length // 3, self.file_length // 2]


    def train(self, train_data, vocab):
        """
        This function builds the word counts and sentence percentages used for classify_text
        train_data: vectorized text
        train_labels: vectorized labels
        vocab: vocab from build_vocab
        """

        # init word counts with 1 due to Dirichlet priors
        num_positive = 0
        num_negative = 0

        for data in train_data:
            # get the last element which is the label
            label = data[-1]
            if label == 1:
                num_positive += 1
            else:
                num_negative += 1

            # iterate through the words in the sentence and add them to the word counts
            for i, word in enumerate(vocab):
                if label == 1:
                    self.positive_word_counts[word] = self.positive_word_counts.get(word, 0) + data[i]
                else:
                    self.negative_word_counts[word] = self.negative_word_counts.get(word, 0) + data[i]

        # calculate the percentage of positive and negative sentences
        self.percent_positive_sentences = num_positive / len(train_data)
        self.percent_negative_sentences = num_negative / len(train_data)
        return 1

    def classify_text(self, vectors, vocab):
        """
        vectors: [vector1, vector2, ...]
        predictions: [0, 1, ...]
        """
        predictions = []
        
        for vector in vectors:
            # init the probability of the sentence being positive or negative
            positive_prob = np.log(self.percent_positive_sentences)
            negative_prob = np.log(self.percent_negative_sentences)

            for i, word in enumerate(vocab):
                if vector[i] == 1: # if the word is in the sentence
                    # calculate the probability of the word being positive or negative
                    positive_prob += np.log((self.positive_word_counts.get(word, 0) + 1) / (self.percent_positive_sentences + len(vocab)))
                    negative_prob += np.log((self.negative_word_counts.get(word, 0) + 1) / (self.percent_negative_sentences + len(vocab)))
                    # total_positive_words = sum(self.positive_word_counts.values())
                    # total_negative_words = sum(self.negative_word_counts.values())

                    # positive_prob += np.log((self.positive_word_counts.get(word, 0) + 1) / (total_positive_words + len(vocab)))
                    # negative_prob += np.log((self.negative_word_counts.get(word, 0) + 1) / (total_negative_words + len(vocab)))

            predictions.append(1 if positive_prob > negative_prob else 0)

        return predictions