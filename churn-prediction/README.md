# Churn Prediction Example

In this example project, we will develop a churn prediction model. We will generate the features by running aggregations on the raw event log data.

What you are going to learn:
- Feature Aggregation
- Log training parameters and metrics
- Model Training
---

To run it, first install Layer SDK:

```
pip install layer-sdk
```

Login to Layer:

```
layer login
```

Check out this example:

```
layer clone https://github.com/layerml/examples.git
cd examples/churn-prediction
```

And, now you are ready to run the project:

```
layer start
```
## File Structure

```yaml
.
|____.layer
| |____project.yaml
|____models
| |____churn_prediction
| | |____model.yaml
| | |____requirements.txt
| | |____model.py
|____README.md
|____data
| |____event_log
| | |____dataset.yaml
| |____user_features
| | |____is_churned.sql
| | |____count_help_view.sql
| | |____count_error.sql
| | |____gender.sql
| | |____count_thumbs_up.sql
| | |____count_login.sql
| | |____count_thumbs_down.sql
| | |____dataset.yaml


```