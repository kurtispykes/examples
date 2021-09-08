# Titanic Survival Example

A classification example with `sklearn.RandomForestClassifier` for predicting the survivals of the Titanic passengers. We will be using the famous [Kaggle Titanic](https://www.kaggle.com/c/titanic/data?select=train.csv) dataset.

## What we are going to learn?

- Feature Store: We are going to use SQL queries to build the `passenger` features.
- Load `passenger` features and use it to train our `survival` model
- Experimentation tracking with
    - logging `accuracy` metric
    - logging `n_estimators` parameter

## Installation & Running

To check out the Layer Titanic Survival example, run:

```bash
layer clone https://github.com/layerml/examples
cd examples/titanic/
```

To run the project:

```bash
layer run
```

## File Structure

```yaml
.
├── .layer
├── data
│   ├── passenger_features	                # feature definitions
│   │   ├── ageband.sql				# Age Band of the passenger
│   │   ├── embarked.sql  			# Embarked or not
│   │   ├── fareband.sql			# Fare Band of the passenger
│   │   ├── is_alone.sql			# Is Passenger travelling alone
│   │   ├── sex.sql				# Sex of the passenger
│   │   ├── survived.sql 			# Survived or not
│   │   ├── title.sql				# Title of the passenger
│   │   └── dataset.yaml				# Declares the metadata of the features above
│   └── titanic_data
│       └── dataset.yaml				# Declares where our source `titanic` dataset is
├── models
│   └── survival_model
│       ├── model.yaml				# Training directives of our model
│       ├── model.py				# Source code of the `Survival` model
│       └── requirements.txt		        # Environment config file
└── README.md
```

