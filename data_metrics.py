import pandas as pd

accuracy_file = './accuracy.csv'

df = pd.read_csv(accuracy_file)
df['accuracy'] = pd.to_numeric(df['video-llava accuracy'], errors='coerce')
overall_accuracy = df['video-llava accuracy'].mean()

print(f'Overall Accuracy: {overall_accuracy:.2f}')