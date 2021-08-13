import pandas as pd
from typing import Any
from layer import Dataset

def build_feature(sdf: Dataset("house_prices_train")) -> Any:
    df = sdf.to_pandas()
    pool_area = df[['PoolArea','Id']]
    return pool_area