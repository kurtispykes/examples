from typing import Any
from layer import Dataset
from sklearn.preprocessing import LabelEncoder


def build_feature(sdf: Dataset("spam_messages")) -> Any:
    df = sdf.to_pandas()
    df = df.sample(10, random_state=0)
    feature_data = df[["id", "label"]]

    # creating instance of labelencoder
    labelencoder = LabelEncoder()
    feature_data = feature_data.assign(is_spam=labelencoder.fit_transform(feature_data["label"]))
    feature_data.drop(columns=["label"], inplace=True)

    return feature_data
