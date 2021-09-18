# Fraud Detection Example

A gradient boosting example with `xgboost` library to reveal suspicious transactions. We will be working with a transaction log dataset from [Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/ntnu-testimon/paysim1).

For experimentation tracking, we will use the area under the precision-recall curve (AUPRC) rather than the conventional area under the receiver operating characteristic (AUROC), since the data is highly skewed.

## What we are going to learn?

- Feature Store: We are going to use SQL queries to build the `transaction` features.
- Load `transaction` features and use it to train the `fraud_detection_model`
- Experimentation tracking
 - logging `auprc` metric
 - logging parameters: `max_depth` and `objective`

## Installation & Running

To check out the Layer Fraud Detection example, run:

```bash
layer clone https://github.com/layerml/examples.git
cd examples/fraud-detection
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
│   ├── transaction_features		# feature definitions
│   │   ├── error_balance_dest.sql	# Error Balance on the destination account
│   │   ├── error_balance_orig.sql	# Error Balance on the originating account
│   │   ├── is_fraud.sql		# Our target value
│   │   ├── new_balance_dest.sql	# New Balance on the destination account
│   │   ├── new_balance_orig.sql	# New Balance on the originating account
│   │   ├── old_balance_dest.sql	# Old Balance on the destination account
│   │   ├── old_balance_orig.sql	# Old Balance on the originating account
│   │   ├── type.sql		        # Type of the transaction feature
│   │   └── dataset.yml			# Declares the metadata of the features above
│   └── transaction
│       └── dataset.yml			# Declares where our source `transactions` dataset is
├── models
│   └── fraud_detection_model
│       ├── model.py			# Source code of the fraud detection model
│       ├── model.yml			# Training directives of our model
│       └── requirements.txt		# Environment config file
└── README.md
```
