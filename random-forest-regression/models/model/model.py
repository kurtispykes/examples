"""House Price
This file demonstrates how we can develop and train our random_forest_regressor by using the
`features` we've developed earlier. Every ML random_forest_regressor project
should have a definition file like this one.
"""
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from layer import Featureset, Train


def train_model(train: Train, pf: Featureset("house_features")) -> Any:
    """Model train function
    This function is a reserved function and will be called by Layer
    when we want this random_forest_regressor to be trained along with the parameters.
    Just like the `features` featureset, you can add more
    parameters to this method to request artifacts (datasets,
    featuresets or models) from Layer.
    Args:
        train (layer.Train): Represents the current train of the random_forest_regressor, passed by
            Layer when the training of the random_forest_regressor starts.
        pf (spark.DataFrame): Layer will return all features inside the
            `features` featureset as a spark.DataFrame automatically
            joining them by primary keys, described in the dataset.yml
    Returns:
       random_forest_regressor: Trained random_forest_regressor object
    """
    # We create the training and label data
    train_df = pf.to_pandas()
    X = train_df.drop(["Id", "SalePrice"], axis=1)
    y = train_df["SalePrice"]

    random_state = 25
    test_size = 0.3
    train.log_parameter("random_state", random_state)
    train.log_parameter("test_size", test_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state)

    # Here we register input & output of the train. Layer will use
    # this registers to extract the signature of the credit_approval_model_bayesian_search and calculate
    # the drift
    train.register_input(X_train)
    train.register_output(y_train)

    n_estimators = 100

    random_forest = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state
    )

    random_forest.fit(X_train, y_train)

    # making predictions
    y_preds = random_forest.predict(X_test)
    mse = mean_squared_error(y_test, y_preds)
    train.log_metric("MSE", mse)

    # We return the credit_approval_model_bayesian_search
    return random_forest