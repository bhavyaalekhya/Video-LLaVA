import pandas as pd

accuracy_file = './accuracy.csv'

df = pd.read_csv(accuracy_file)

df['video-llava - accuracy'] = pd.to_numeric(df['video-llava - accuracy'], errors='coerce')

# Calculate the mean of the accuracy values, ignoring NaN values
overall_accuracy = df['video-llava - accuracy'].mean()

print(f'Overall Accuracy: {overall_accuracy:.2f}')