# CS331 Sentiment Analysis Assignment 3
# This file contains the processing functions

import numpy as np
from classifier import BayesClassifier

# Chars to be replaced with spaces
spaceChars = ["/", "-", "_", "+", "="]

# Chars to be removed (replaced with "")
removeChars = [ "'", ",", ":", ";", ".", "!", "?", "(", ")", "[", "]", "{", "}", "*", "&", "^", "%", "$", "#", "@", "~", "`", "|", "\\", "<", ">", "\n" ]

validOneCharWords = ["a", "i"]
validTwoCharWords = ["am", "an", "as", "at", "be", "by", "do", "go", "he", "hi", "if", "in", "is", "it", "me", "my", "no", "of", "oh", "on", "or", "so", "to", "up", "us", "we"]

contractions = {
    "dont": "do not",
    "doesnt": "does not",
    "didnt": "did not",
    "wont": "will not",
    "cant": "can not",
    "couldnt": "could not",
    "shouldnt": "should not",
    "wouldnt": "would not",
    "wouldve": "would have",
    "isnt": "is not",
    "arent": "are not",
    "wasnt": "was not",
    "werent": "were not",
    "hasnt": "has not",
    "havent": "have not",
    "hadnt": "had not",
    "im": "i am",
    "ive": "i have",
    "youre": "you are",
    "youve": "you have",
    "youd": "you would",
    "youll": "you will",
    "theyre": "they are",
    "theyve": "they have",
    "weve": "we have",
    "weve": "we have"
}

"""
Preprocessing Steps:
1. Strip punctuation:
    Remove all punctuation marks from the text. Characters such as slashes should be replaced with spaces. Characters such as apostrophes should be removed.
    Convert letters to lowercase.
"""
def process_text(text):
    # preprocessed_text is a list of tuples (text, label)
    preprocessed_text = []

    # for each list in text
    for line in text:
        line = line.lower()

        for char in line:
            if char in removeChars:
                line = line.replace(char, "")
            elif char in spaceChars:
                line = line.replace(char, " ")

        newTuple = line.split("\t")
        newTuple[1] = int(newTuple[1])

        # remove the space from the end of newTuple[0] using slicing
        newTuple[0] = newTuple[0][:-1]

        preprocessed_text.append(newTuple)

    return preprocessed_text

"""
Builds the vocab from the preprocessed text
preprocessed_text: output from process_text
Returns unique text tokens
"""
def build_vocab(preprocessed_text):
    # preprocessed_text is a list of tuples (text, label)

    # vocab will be an alphabetical list of all unique words in each preprocessed_text[i][0]
    vocab = []

    # Working non-alphabetical list of all unique words in each preprocessed_text[i][0]
    for line in preprocessed_text:
        words = line[0].split(" ")
        for word in words:
            # check if the current word is a number or contains a number
            if word.isnumeric() or any(char.isdigit() for char in word):
                continue

            # check if the current word is a contraction and expand it
            if word in contractions:
                # expanded: ['i', 'am']
                expanded = contractions[word].split(" ")
                # if the two words are not already in vocab, add them
                if expanded[0] not in vocab:
                    vocab.append(expanded[0])
                if expanded[1] not in vocab:
                    vocab.append(expanded[1])

                continue

            # check if the current word is a valid one or two character word
            if len(word) == 1:
                if word not in validOneCharWords:
                    continue
            elif len(word) == 2:
                if word not in validTwoCharWords:
                    continue

            if word not in vocab:
                vocab.append(word)

    vocab.sort()

    # this function sometimes adds an empty string to the beginning of the list for some reason
    if vocab[0] == "":
        vocab.pop(0)

    return vocab

"""
Converts the text into vectors
text: preprocess_text from process_text
vocab: vocab from build_vocab
Returns the vectorized text and the labels
"""
def vectorize_text(text, vocab):
    vectorized_text = []

    for line in text:
        # init a feature vector as a list of 0s, one for each word in vocab
        feature_vector = [0] * len(vocab)

        # split the line into words
        words = line[0].split(" ")

        for word in words:
            if word in vocab:
                feature_vector[vocab.index(word)] = 1

        # append the class label
        feature_vector.append(line[1])

        vectorized_text.append(feature_vector)

    return vectorized_text


def accuracy(predicted_labels, true_labels):
    """
    predicted_labels: list of 0/1s predicted by classifier
    true_labels: list of 0/1s from text file
    return the accuracy of the predictions
    """

    # accuracy_score is the number of correct predictions divided by the total number of predictions
    accuracy_score = 0

    for i in range(len(predicted_labels)):
        if predicted_labels[i] == true_labels[i]:
            accuracy_score += 1

    accuracy_score /= len(predicted_labels)

    return accuracy_score

def output_preprocessed(filename, vocab, vectorized_data):
    with open(filename, "w") as f:
        f.write(",".join(vocab) + ",classlabel\n")

        for line in vectorized_data:
            f.write(",".join(str(x) for x in line) + "\n")

def main():
    training_data = []
    testing_data = []

    # Read each line of the training data in to its own list and append to training_data
    with open("trainingSet.txt", "r") as f:
        for line in f:
            training_data.append(line)

    # Read each line of the testing data in to its own list and append to testing_data
    with open("testSet.txt", "r") as f:
        for line in f:
            testing_data.append(line)

    print("== Read in training and testing data")

    # Preprocess the training and testing data
    processed_training_data = process_text(training_data)
    processed_testing_data = process_text(testing_data)
    print("== Preprocessed training and testing data")

    training_vocab = build_vocab(processed_training_data)
    print("== Built vocab")

    # Vectorize the training and testing data
    vectorized_training_data = vectorize_text(processed_training_data, training_vocab)
    vectorized_testing_data = vectorize_text(processed_testing_data, training_vocab)
    print("== Vectorized training and testing data")

    # Output the preprocessed training and testing data
    print("== Outputting preprocessed training and testing data")
    output_preprocessed("preprocessed_train.txt", training_vocab, vectorized_training_data)
    output_preprocessed("preprocessed_test.txt", training_vocab, vectorized_testing_data)

    model = BayesClassifier()

    # Divide the vectorized_training_data into 4 equal parts
    length = len(vectorized_training_data)
    quarter = length // 4

    # Train the model on the first 1/4 of the vectorized_training_data, then 1/2, then 3/4, then all of it.
    # After each training, test the model on the vectorized_testing_data and print the accuracy

    # with open('results.txt', 'w') as f:
    #     f.write("=== Results ===\n")

    #     for i in range(1, 5):
    #         model.train(vectorized_training_data[0:i * quarter], training_vocab)
    #         print("== Trained model on", i * quarter, "lines of data")

    #         print("== Model Info")
    #         print(" -- model.percent_positive_sentences", model.percent_positive_sentences)
    #         print(" -- model.percent_negative_sentences", model.percent_negative_sentences)

    #         predictions = model.classify_text(vectorized_testing_data, training_vocab)
    #         print("== Tested model on", len(vectorized_testing_data), "lines of data")
    #         print(" -- Accuracy:", accuracy(predictions, [x[-1] for x in vectorized_testing_data]))

    #         f.write("== Training model on " + str(i * quarter) + " lines of data\n")
    #         f.write("== Trained model on " + str(i * quarter) + " lines of data\n")
    #         f.write("== Model Info\n")
    #         f.write(" -- model.percent_positive_sentences " + str(model.percent_positive_sentences) + "\n")
    #         f.write(" -- model.percent_negative_sentences " + str(model.percent_negative_sentences) + "\n")
    #         f.write("== Testing model on " + str(len(vectorized_testing_data)) + " lines of data\n")
    #         f.write("== Tested model on " + str(len(vectorized_testing_data)) + " lines of data\n")
    #         f.write(" -- Accuracy: " + str(accuracy(predictions, [x[-1] for x in vectorized_testing_data])) + "\n")

    # Train the model on all of the vectorized_training_data
    model.train(vectorized_training_data, training_vocab)
    print("== Trained model on all", len(vectorized_training_data), "lines of data")

    print("== Model Info")
    print(" -- model.percent_positive_sentences", model.percent_positive_sentences)
    print(" -- model.percent_negative_sentences", model.percent_negative_sentences)
    # print(" -- model.positive_word_counts", model.positive_word_counts)
    # print(" -- model.negative_word_counts", model.negative_word_counts)

    predictions = model.classify_text(vectorized_testing_data, training_vocab)
    print("== Tested model on", len(vectorized_testing_data), "lines of data")
    print(" -- Accuracy:", accuracy(predictions, [x[-1] for x in vectorized_testing_data]))

    return 1

if __name__ == "__main__":
    main()