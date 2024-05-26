import pandas as pd

metrics_file = './metrics.csv'

metrics = pd.read_csv(metrics_file)

print(metrics)