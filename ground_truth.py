import os
import json
import pandas as pd
from tqdm import tqdm

def ground_truth(name, video, normal_annot, questions):
    gt = []
    steps = video['steps']
    normal = name + '_x'
    n_steps = normal_annot[normal]['steps']
    
    # Creating a mapping from step descriptions to their index in the questions list
    question_mapping = {question['q'].split(' ', 1)[1].split('?')[0].strip(): i for i, question in enumerate(questions)}
    
    # Initializing ground truth with 0s
    gt = [0] * len(questions)
    
    for step in steps:
        step_desc = step['description'].split(' -', 1)[-1].strip()  # Extracting the step description
        for q_desc, q_index in question_mapping.items():
            if q_desc in step_desc and step['has_errors'] and "Preparation Error" in step['errors']:
                gt[q_index] = 1
    
    return gt

def question_index(related_questions):
    question_to_index = {}
    index_counter = 0
    for question in related_questions:
        question_to_index[question] = index_counter
        index_counter += 1
    return question_to_index

def flatten(l):
    return [label for sublist in l for label in sublist]

def open_file(filename):
    with open(filename, 'r') as file:
        contents = json.load(file)
    
    return contents

def error_gt(video_dir, q_file, normal_annot, steps, error_type):
    qs = open_file(q_file)
    n_annot = open_file(normal_annot)
    step_annot = open_file(steps)
    
    g_truth = []

    for v in tqdm(os.listdir(video_dir), desc="Processing videos"):
        if '.DS_Store' not in v:
            name = v.split('_')
            gt_name = name[0] + '_' + name[1]
            g = ground_truth(name[0], step_annot[gt_name], n_annot, qs[name[0]+'_x']['questions'])
            g_truth.append(g)
            pass

    g_truth = flatten(g_truth)
    print(f'{error_type} ground_truth: ', g_truth)

    print(len(g_truth))


def main():
    video_dir = '/Users/bhavyaalekhya/Desktop/Thesis/dataset/resolution_360p'
    m_file = './error_prompts/measurement_error.json'
    normal_annot_file = './normal_videos.json'
    steps = './step_annotations.json'

    print("Preparation error type: ")
    error_gt(video_dir, m_file, normal_annot_file, steps, 'Measurement Error')

if __name__ == "__main__":
    main()
