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

    question_map = {q: i for i, q in enumerate(questions)}

    # Initialize gt with zeros for the length of common steps
    gt = [0] * len(common_steps)

    for step in error_steps:
        if step['description'] in common_steps:
            for error in step.get('errors', []):
                if error['tag'] == 'Order Error':
                    index = common_steps.index(step['description'])
                    gt[index] = 1

    return gt

def ground_truth(name, video, normal_annot, questions):
    gt = []
    steps = video['steps']
    normal = name + '_x'
    n_steps = normal_annot[normal]['steps']
    n_steps_desc = []

    for step in n_steps:
        n_steps_desc.append(step['description'])

    video_steps_desc = [step['description'] for step in steps]
    common_steps = list(set(n_steps_desc).intersection(video_steps_desc))
    q = len(questions)
    followup = [len(j['followup']) for i, j in enumerate(questions)]
    total_length = q + sum(followup)
    
    gt = [0] * total_length

    for step in steps:
        if step['description'] in common_steps:
            index = common_steps.index(step['description'])
            question = questions[index]
            if 'followup' in question.keys():
                if step['has_errors'] and "Order Error" in step['errors']:
                    gt[index] = 1
            else:
                if step['has_errors']:
                    gt[index] = 1

    current_index = q
    for i, question in enumerate(questions):
        if 'followup' in question.keys():
            followup_gt = [0] * len(question['followup'])
            for j, followup in enumerate(question['followup']):
                if followup in video_steps_desc:
                    followup_gt[j] = 1
            gt[current_index:current_index + len(question['followup'])] = followup_gt
            current_index += len(question['followup'])

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
        name = v.split('_')
        gt_name = name[0] + '_' + name[1]
        g = ground_truth(name[0], step_annot[gt_name], n_annot, qs[name[0]+'_x'])
        g_truth.append(g)
        pass

    g_truth = flatten(g_truth)
    print(f'{error_type} ground_truth: ', g_truth)

    output_name = "_".join(error_type.lower().split(" "))

    output_name = './' + output_name + '.txt'

    content = f'Ground Truth: {g_truth}'

    with open(output_name, 'w') as file:
        file.write(content)


def main():
    video_dir = '/data/rohith/captain_cook/videos/gopro/resolution_360p/'
    m_file = './error_prompts/order_error.json'
    normal_annot_file = './normal_videos.json'
    steps = './step_annotations.json'

    print("Order error type: ")
    error_gt(video_dir, m_file, normal_annot_file, steps, 'Order Error')

if __name__ == "__main__":
    main()
