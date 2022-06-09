import os
import string
import pandas as pd
import warnings
import random
import stopwordsiso as stopwords
import io
import json
import numpy as np
import matplotlib.pyplot as plt
#from wordcloud import WordCloud
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Embedding, Dropout
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers.convolutional import Conv1D
from tensorflow.keras.utils import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.metrics import Precision, Recall
#from cube.api import Cube

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=FutureWarning)


def generate_random_review(dataframe):
    # Selects a random review from the dataset
    rows = len(dataframe.index)
    rows = rows - 1
    review_row = random.randrange(0, rows)
    return dataframe['ReviewText'].iloc[review_row]


def preprocess_review(review):
    # Preprocesses a review for live prediction
    stopword_list = r'\b(?:{})\b'.format('|'.join(stopwords.stopwords("ro")))

    review = review.replace('[{}]'.format(string.punctuation), '')
    review = review.replace('[{}]'.format(string.digits), '')
    review = review.replace(stopword_list, '')
    review = review.replace(r'\s+', ' ')
    review = review.lower()
    return review


def predict_rating(review, tokenizer, model):
    # Returns the rating of a live predicted review
    review = preprocess_review(review)
    tokenized_review = tokenizer.texts_to_sequences([review])
    pad_len = 500
    padded_review = pad_sequences(tokenized_review, maxlen=pad_len)
    predict = model.predict(padded_review)
    classes = np.argmax(predict, axis=-1)
    return classes[0]


def remove_punctuation(dataframe):
    # Removes the entire punctuation from the dataset
    dataframe['ReviewText'] = dataframe['ReviewText'].str.replace('[{}]'.format(string.punctuation), '')
    return dataframe


def remove_digits(dataframe):
    # Removes all the digits from the dataset
    dataframe['ReviewText'] = dataframe['ReviewText'].str.replace('[{}]'.format(string.digits), '')
    return dataframe


def make_lowercase(dataframe):
    # Makes all reviews in the dataset in lowercase
    dataframe['ReviewText'] = dataframe['ReviewText'].str.lower()
    return dataframe


def remove_stopwords(dataframe):
    # Removes stopwords from the dataset
    stopword_list = r'\b(?:{})\b'.format('|'.join(stopwords.stopwords("ro")))
    dataframe['ReviewText'] = dataframe['ReviewText'].str.replace(stopword_list, '')
    dataframe['ReviewText'] = dataframe['ReviewText'].str.replace(r'\s+', ' ')
    return dataframe


def sentiment_encode(df, column, le):
    # One hot encodes the label column of the dataset
    sentiment_le = le.fit_transform(df[column].to_numpy().reshape(-1, 1))
    return sentiment_le, le


def preprocess_data(dataframe):
    # Performs all the pre-processing steps
    dataframe = remove_punctuation(dataframe)
    dataframe = remove_digits(dataframe)
    dataframe = remove_stopwords(dataframe)
    dataframe = make_lowercase(dataframe)
    #dataframe = lemmatize_data(dataframe)

    return dataframe


def build_model():
    # Creates the model to be used for training. Layers can be added below
    model = Sequential()

    filters = 128
    kernel_size = 4

    model.add(Embedding(vocab_size, embedding_dims, input_length=max_len))
    model.add(Conv1D(filters, kernel_size=kernel_size))
    model.add(Dropout(0.5))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(noutput, activation='softmax'))

    return model


if os.path.exists("Scraped reviews.xlsx"):
    df = pd.read_excel(r'Scraped reviews.xlsx')

    # The following block will plot the rating distribution of the dataset if uncommented
    # df['Rating'].value_counts(sort=False).sort_index().plot.bar()
    # plt.xticks(rotation='horizontal')
    # plt.show()

    # The following block will plot a word cloud of the most important features in the dataset if uncommented
    # df = preprocess_data(df)
    # text = ' '.join(df['ReviewText'])
    # wordcloud2 = WordCloud().generate(text)
    # plt.imshow(wordcloud2)
    # plt.axis("off")
    # plt.show()

    if os.path.isdir("cnn_model"):
        model = load_model("cnn_model")
        # print(model.summary())
    else:
        df = preprocess_data(df)
        max_features = 5000
        oov = "OOV"
        tokenizer = Tokenizer(num_words=max_features, oov_token=oov)
        tokenizer.fit_on_texts(df["ReviewText"])
        tokenizer_json = tokenizer.to_json()

        # The tokenizer is saved so we don't have to train it every time unless we train a new model
        with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))
        tokenized = tokenizer.texts_to_sequences(df["ReviewText"])
        le = OneHotEncoder()
        sentiment_le, le = sentiment_encode(df, "Rating", le)

        max_len = 500
        Xtrain = pad_sequences(tokenized, maxlen=max_len)

        # Here are the most important parameters for the training function
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        vocab_size = max_features
        embedding_dims = 128
        num_epochs = 10
        noutput = 5
        fold_nr = 1

        max_accuracy = 0
        average_accuracy = 0
        # We want to keep track of precision and recall per rating class for every fold, as well as overall accuracy
        average_precision = [0, 0, 0, 0, 0]
        average_recall = [0, 0, 0, 0, 0]
        for train, test in kfold.split(Xtrain, sentiment_le.toarray().argmax(1)):
            print("Fold number: ", fold_nr)
            model = build_model()
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy',
                                                                                      Precision(class_id=0),
                                                                                      Precision(class_id=1),
                                                                                      Precision(class_id=2),
                                                                                      Precision(class_id=3),
                                                                                      Precision(class_id=4),
                                                                                      Recall(class_id=0),
                                                                                      Recall(class_id=1),
                                                                                      Recall(class_id=2),
                                                                                      Recall(class_id=3),
                                                                                      Recall(class_id=4)])
            model.fit(Xtrain[train], sentiment_le.toarray()[train], epochs=num_epochs,
                      batch_size=64,
                      validation_data=(Xtrain[test], sentiment_le.toarray()[test]),
                      verbose=1)
            results = model.evaluate(Xtrain[test], sentiment_le.toarray()[test])
            print(results)
            print("Test loss: %.2f" % results[0])
            print("Test accuracy: %.2f%%" % (results[1] * 100))

            average_accuracy = average_accuracy + results[1] * 100
            for index, precision in enumerate(average_precision):
                precision = precision + results[index + 2]
            for index, recall in enumerate(average_recall):
                recall = recall + results[index + 7]
            if results[1] * 100 > max_accuracy:
                model.save("cnn_model")
            fold_nr = fold_nr + 1
        average_accuracy = average_accuracy / (fold_nr - 1)
        print("Average accuracy: %.2f%%" % average_accuracy)
        for precision in average_precision:
            precision = precision / (fold_nr - 1)
        for recall in average_recall:
            recall = recall / (fold_nr - 1)

        for i in range(1, 6):
            print("Rating " + str(i) + " - Avg Precision: %.2f%% " % (results[i + 1] * 100) + "   Recall: %.2f%%" % (
                    results[i + 6] * 100))

else:
    print("Scrape reviews first")
