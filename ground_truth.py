import os
import json
import pandas as pd
from tqdm import tqdm

def gt(name, video, error_annot, normal_annot, questions, error_type):
    gt = []
    steps = video['steps']
    error_steps = error_annot['step_annotations']
    normal = name + '_x'
    n_steps = normal_annot[normal]['steps']
    n_steps_desc = [step['description'] for step in n_steps]

    video_steps_desc = [step['description'] for step in steps]
    common_steps = list(set(n_steps_desc).intersection(video_steps_desc))

    # Initialize gt with zeros for the length of common steps
    gt = [0] * len(common_steps)

    for step in error_steps:
        if step['description'] in common_steps:
            for error in step.get('errors', []):
                if error['tag'] == error_type:
                    index = common_steps.index(step['description'])
                    gt[index] = 1

    return gt

def question_index(related_questions):
    question_to_index = {}
    index_counter = 0
    for question in related_questions:
        question_to_index[question['q']] = index_counter
        if 'followup' in question.keys():
            for followup in question['followup']:
                question_to_index[followup] = index_counter
        index_counter += 1
    return question_to_index

def flatten(l):
    return [label for sublist in l for label in sublist]

def open_file(filename):
    with open(filename, 'r') as file:
        contents = json.load(file)
    
    return contents

def error_gt(video_dir, q_file, error_annot, normal_annot, steps, error_type):
    qs = open_file(q_file)
    gt_f = open_file(error_annot)
    n_annot = open_file(normal_annot)
    step_annot = open_file(steps)
    
    g_truth = []

    for v in tqdm(os.listdir(video_dir), desc="Processing videos"):
        video = os.path.join(video_dir, v)
        name = v.split("_")
        gt_name = name[0] + '_' + name[1]
        related_questions = qs[name[0] + "_x"]["questions"]
        for idx, entry in enumerate(gt_f):
            if entry['recording_id']==gt_name:
                g_t = gt(name[0], step_annot[gt_name], gt_f[idx], n_annot, related_questions, error_type)
                g_truth.append(g_t)

        question_ind = question_index(related_questions)

    g_truth = flatten(g_truth)
    print(f'{error_type} ground_truth: ', g_truth)

    output_name = "_".join(error_type.lower().split(" "))

    output_name = './' + output_name + '.txt'

    content = f'Ground Truth: {g_truth}'

    with open(output_name, 'w') as file:
        file.write(content)


def main():
    video_dir = '/data/rohith/captain_cook/videos/gopro/resolution_360p/'
    m_file = './error_prompts/missing_error.json'
    o_file = './error_prompts/order_error.json'
    temp_file = './error_prompts/temperature_error.json'
    error_annot_file = './error_annotations.json'
    normal_annot_file = './normal_videos.json'
    steps = './step_annotations.json'

    #print("Missing error type: ")
    #error_gt(video_dir, m_file, error_annot_file, normal_annot_file, steps, 'Missing Error')

    print("Order error type: ")
    error_gt(video_dir, o_file, error_annot_file, normal_annot_file, steps, 'Order Error')

    print("Temperature error type: ")
    error_gt(video_dir, temp_file, error_annot_file, normal_annot_file, steps, 'Temperature Error')

if __name__ == "__main__":
    
    main()
