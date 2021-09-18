# Spam Detection Project Example

An NLP example with `nltk` library to predict the spam SMS messages. In this project, we are going to use Python Features to remove stop words and to lemmatize messages. Also, we are going to load an ML model from the Layer Model Catalog to create training data for the `spam_detection` model.

## What we are going to learn?

- Extract advanced features from our data with Python Features utilizing `nltk` and `scikit` libraries.
- Use a model to create a training data for another model
- Experimentation tracking with logging metrics: `f1_score`, `accuracy` and `mean_scores`

## Installation & Running

To check out the Layer Spam Detection example, run:

```bash
layer clone https://github.com/layerml/examples.git
cd examples/spam-detection
```

To run the project:

```bash
layer start
```

## File Structure

```yaml
.
├── .layer
├── data
│   ├── sms_featureset
│   │   ├── is_spam
│   │   │   ├── feature.py			# Source code of the `is_spam` feature. We do basic labelencoding.
│   │   │   ├── requirements.txt	        # Environment config file for the `is_spam` feature
│   │   ├── message
│   │   │   ├── feature.py			# Source code of the `message` feature. We remove stop words and lemmatize messages.
│   │   │   ├── requirements.txt	        # Environment config file for the `message` feature
│   │   └── dataset.yml
│   └── spam_data
│       └── dataset.yml				# Declares where our source `spam_messages` dataset is
├── models
│   └── vectorizer
│       ├── model.yml				# Training directives of our model
│       ├── model.py				# Source code of the `Vectorizer` model
│       └── requirements.txt		        # Environment config file
│   └── spam_detection
│       ├── model.yml				# Training directives of our model
│       ├── model.py				# Source code of the `Spam Detection` model
│       └── requirements.txt		        # Environment config file
└── README.md
```

