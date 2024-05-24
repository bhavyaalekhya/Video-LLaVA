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

df2 = pd.read_csv(precision_file)

df2['video-llava - precision'] = pd.to_numeric(df2['video-llava - precision'], errors='coerce')

df3 = pd.read_csv(recall_file)

df3['video-llava - recall'] = pd.to_numeric(df3['video-llava - recall'], errors='coerce')

df4 = df1.append(df2).append(df3)

tp = 0
fp = 0
fn = 0

for idx, row in df4.iterrows():
    tp_i = (row['video-llava f1_score'] * (row['video-llava precision'] + row['video-llava recall']))/(2*row['video-llava precision']*row['video-llava recall'])

    fp_i = (tp_i) * ((1/row['video-llava precision'])-1)

    fn_i = (tp_i) * ((1/row['video-llava recall'])-1)

    tp += tp_i
    fp += fp_i
    fn += fn_i

overall_precision = tp / (fp+tp)
overall_recall = tp / (tp + fn)
overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall)

print(f'Overall Accuracy: {overall_accuracy:.2f}')
print(f'Overall F1: {overall_f1:.2f}')
print(f'Overall Precision: {overall_precision:.2f}')
print(f'Overall Recall: {overall_recall:.2f}')