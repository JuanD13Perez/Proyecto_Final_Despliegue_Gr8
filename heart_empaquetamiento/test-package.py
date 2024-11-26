import pandas as pd
import numpy as np
from model.predict import make_prediction

sample_input_data = pd.read_csv("~/test/heart_analytics_test.csv")
result = make_prediction(input_data=sample_input_data)
print(result)