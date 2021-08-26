"""New Project Example
This file demonstrates how we can develop and train our spam_detection by using the
`sms_features` we've developed earlier. Every ML spam_detection project
should have a definition file like this one.
"""
from typing import Any
from layer import Featureset, Train

from sklearn.model_selection import train_test_split
import wget
import zipfile
from  tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense,GlobalAveragePooling1D,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

def train_model(train: Train, pf: Featureset("word_embedding_featureset")) -> Any:
    """Model train function
    This function is a reserved function and will be called by Layer
    when we want this spam_detection to be trained along with the parameters.
    Just like the `sms_features` featureset, you can add more
    parameters to this method to request artifacts (datasets,
    featuresets or models) from Layer.
    Args:
        train (layer.Train): Represents the current train of the spam_detection, passed by
            Layer when the training of the spam_detection starts.
        pf (spark.DataFrame): Layer will return all sms_features inside the
            `sms_features` featureset as a spark.DataFrame automatically
            joining them by primary keys, described in the dataset.yml
    Returns:
       spam_detection: Trained spam_detection object
    """
    df = pf.to_pandas()
    X = df['message']
    y = df['is_spam']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=0
    )
    train.register_input(X_train)
    train.register_output(y_train)

    vocab_size = 1000
    oov_token = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(X_train)

    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    max_length = 100
    padding_type = "post"
    truncation_type = "post"

    X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding=padding_type,
                                   truncating=truncation_type)
    X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length,
                                  padding=padding_type, truncating=truncation_type)


    url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    filename = wget.download(url)

    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('glove')

    word_index = tokenizer.word_index

    embeddings_index = {}
    f = open('glove/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((len(word_index) + 1, max_length))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1,
                                max_length,
                                weights=[embedding_matrix],
                                input_length=max_length,
                                trainable=False)


    model = Sequential([
        embedding_layer,
        Dropout(0.2),
        GlobalAveragePooling1D(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                                 loss=keras.losses.BinaryCrossentropy(from_logits=True),
                                 metrics=[keras.metrics.BinaryAccuracy()])

    callbacks = [
        EarlyStopping(patience=10),
    ]
    num_epochs = 600
    model.fit(X_train_padded, y_train, epochs=num_epochs, validation_data=(X_test_padded, y_test),
                        callbacks=callbacks)

    loss, accuracy = model.evaluate(X_train_padded, y_train, verbose=1)

    train.log_metric("Training Accuracy", accuracy)
    train.log_metric("Training Loss", loss)

    test_loss, test_accuracy = model.evaluate(X_test_padded, y_test)
    train.log_metric("Testing Accuracy", test_accuracy)
    train.log_metric("Testing Loss", test_loss)

    return model