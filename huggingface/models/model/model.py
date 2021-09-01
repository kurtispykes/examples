from typing import Any

from layer import Featureset, Train
from sklearn.model_selection import train_test_split
import tensorflow as tf

from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification


def train_model(train: Train, pf: Featureset("hugging_features")) -> Any:
    """Model train function
    This function is a reserved function and will be called by Layer
    when we want this model to be trained along with the parameters.
    Just like the `features` featureset, you can add more
    parameters to this method to request artifacts (datasets,
    featuresets or models) from Layer.
    Args:
        train (layer.Train): Represents the current train of the model, passed by
            Layer when the training of the model starts.
        pf (spark.DataFrame): Layer will return all features inside the
            `features` featureset as a spark.DataFrame automatically
            joining them by primary keys, described in the dataset.yml
    Returns:
       model: Trained model object
    """
    df = pf.to_pandas()
    X = list(df['message'])
    y = list(df['is_spam'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True)
    X_val_tokenized = tokenizer(X_test, padding=True, truncation=True)

    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(X_train_tokenized),
        y_train
    ))

    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(X_val_tokenized),
        y_test
    ))

    # train.register_input(X_train_tokenized)
    # train.register_output(y_train)

    model = TFBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    model.fit(train_dataset.shuffle(1000).batch(16), validation_data=test_dataset.shuffle(1000).batch(16), epochs=3,
              batch_size=16)
    loss, accuracy = model.evaluate(train_dataset.shuffle(1000).batch(16))
    train.log_metric("Training Accuracy", accuracy)
    train.log_metric("Training Loss", loss)

    test_loss, test_accuracy = model.evaluate(test_dataset.shuffle(1000).batch(16))
    train.log_metric("Testing Accuracy", test_accuracy)
    train.log_metric("Testing Loss", test_loss)

    return model

