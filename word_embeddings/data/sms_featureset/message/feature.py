from typing import Any
from layer import Dataset


def build_feature(sdf: Dataset("spam_messages")) -> Any:

    df = sdf.to_pandas()
    feature_data = df[["id", "message"]]

    return feature_data


