from typing import Any
from layer import Featureset, Train, Context
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
def train_model(train: Train, context: Context, pf: Featureset("passenger_features")) -> Any:
    df = pf.to_spark()
    ## cols: PassengerId, AgeBand, EmbarkStatus, FareBand, IsAlone, Sex, Survived, Title
    feat_cols = [ 'AgeBand', 'EmbarkStatus', 'FareBand', 'IsAlone', 'Sex', 'Title' ]
    ## ORIG ## feat_cols = [ 'ageband','embarked','isAlone','fareband','sex','title']
    label_col = 'Survived'
    vec_assember = VectorAssembler(inputCols=feat_cols, outputCol='features')
    final_data = vec_assember.transform(df)
    test_size = 0.2
    training_size = 0.8
    train.log_parameter("test_size", test_size)
    seed = 42
    train.log_parameter("seed", seed)
    training, testing = final_data.randomSplit([training_size, test_size], seed=seed)
    lr = LogisticRegression(labelCol=label_col, featuresCol='features')
    lrModel = lr.fit(training)
    predictions = lrModel.transform(testing)
    evaluator = BinaryClassificationEvaluator(labelCol=label_col)
    train.log_metric("BinaryClassificationEvaluator", evaluator.evaluate(predictions))
    return lrModel