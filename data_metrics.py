import pandas as pd

accuracy_file = './accuracy.csv'
f1_score_file = './f1_score.csv'
precision_file = './precision.csv'
recall_file = './recall.csv'

df = pd.read_csv(accuracy_file)

df['video-llava - accuracy'] = pd.to_numeric(df['video-llava - accuracy'], errors='coerce')

# Calculate the mean of the accuracy values, ignoring NaN values
overall_accuracy = df['video-llava - accuracy'].mean()

df1 = pd.read_csv(f1_score_file)

df1['video-llava - f1_score'] = pd.to_numeric(df1['video-llava - f1_score'], errors='coerce')

# Calculate the mean of the accuracy values, ignoring NaN values
overall_f1 = df1['video-llava - f1_score'].mean()

df2 = pd.read_csv(precision_file)

df2['video-llava - precision'] = pd.to_numeric(df2['video-llava - precision'], errors='coerce')

# Calculate the mean of the accuracy values, ignoring NaN values
overall_precision = df2['video-llava - precision'].mean()

df3 = pd.read_csv(recall_file)

df3['video-llava - recall'] = pd.to_numeric(df3['video-llava - recall'], errors='coerce')

# Calculate the mean of the accuracy values, ignoring NaN values
overall_recall = df3['video-llava - recall'].mean()

print(f'Overall Accuracy: {overall_accuracy:.2f}')
print(f'Overall F1: {overall_f1:.2f}')
print(f'Overall Precision: {overall_precision:.2f}')
print(f'Overall Recall: {overall_recall:.2f}')