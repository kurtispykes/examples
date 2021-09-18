# Titanic Survival Example with Hyper Parameter Tuning

This is an example project which focuses development of an ml model to predict the survivals of the passenger in the Titanic disaster. This model will use the features extracted from the popular [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic/data).

What you are going to learn:
- Layer Feature Store: Creating and consuming features
- Hyper parameter tuning
- Model Training in Layer
---

To run it, first install Layer SDK:

```
pip install layer-sdk
```

Login to Layer:

```
layer login
```

To check out the Layer Titanic Survival example, run:

```bash
layer clone https://github.com/layerml/examples
cd examples/titanic/hyperparametertuning
```


And, now you are ready to run the project:

```
$ layer start

Layer v0.18.1
Found 1 featureset + 1 model + 0 tests = 2 entities in total

19:08:44 | 1 of 2 BUILD passenger_features .................. [RUN]
19:08:50 | 1.1 of 2  FEATURE ageband ........................ [SUCCESS 1.1 in 1s]
19:08:51 | 1.2 of 2  FEATURE embarked ....................... [SUCCESS 1.2 in 1s]
19:08:52 | 1.3 of 2  FEATURE isAlone ........................ [SUCCESS 1.3 in 1s]
19:08:53 | 1.4 of 2  FEATURE fareband ....................... [SUCCESS 1.4 in 1s]
19:08:54 | 1.5 of 2  FEATURE sex ............................ [SUCCESS 1.5 in 1s]
19:08:55 | 1.6 of 2  FEATURE title .......................... [SUCCESS 1.6 in 1s]
19:08:56 | 1.7 of 2  FEATURE survival ....................... [SUCCESS 1.7 in 1s]
19:08:57 | 1 of 2 END passenger_features .................... [SUCCESS 1 in 7s]
19:08:57 |
19:08:57 | 2 of 2 TRAIN survival_model ...................... [RUN]
19:09:00 | 2 of 2 END survival_model ........................ [SUCCESS 2 in 3s]
19:09:00 |
19:09:00 | Finished running 1 featureset + 1 model in 16s.

Completed successfully

Done. PASS=2 WARN=0 ERROR=0 SKIP=0 TOTAL=2
```
