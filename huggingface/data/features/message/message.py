from typing import Any
from layer import Dataset


def build_feature(sdf: Dataset("spam_messages")) -> Any:
    df = sdf.to_pandas()
    df = df.sample(10, random_state=0)
    feature_data = df[["id", "message"]]
    return feature_data
