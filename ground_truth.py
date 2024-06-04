import os
import json
import pandas as pd
from tqdm import tqdm

def gt(name, video, error_annot, normal_annot, questions):
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

    # Mapping question descriptions to their indices
    question_map = {q['q']: i for i, q in enumerate(questions)}

    for step in error_steps:
        if step['description'] in common_steps:
            if step['is_error']:
                for error in step.get('errors', []):
                    if error['tag'] == "Preparation Error":
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

def data_file(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, sep=',', mode='a+')

def main():
    video_dir = '/data/rohith/captain_cook/videos/gopro/resolution_360p/'
    questions_file = './error_prompts/preparation_error.json'
    error_annot_file = './error_annotations.json'
    normal_annot_file = './normal_videos.json'

    with open(questions_file, 'r') as f:
        qs = json.load(f)

    with open(error_annot_file, 'r') as file:
        gt_f = json.load(file)

    with open(normal_annot_file, 'r') as f:
        n_annot = json.load(f)
    
    g_truth = []

    for v in tqdm(os.listdir(video_dir), desc="Processing videos"):
        if v=='1_28_360p.mp4':
            video = os.path.join(video_dir, v)
            name = v.split("_")
            gt_name = name[0] + '_' + name[1]
            related_questions = qs[name[0] + "_x"]["questions"]
            for idx, entry in enumerate(gt_f):
                if entry['recording_id']=='1_28':
                    g_t = gt(name[0], gt_f[idx], gt_f[idx], n_annot, related_questions)
                    g_truth.append(g_t)

            question_ind = question_index(related_questions)

    g_truth = flatten(g_truth)
    print('ground_truth: ', g_truth)

if __name__ == "__main__":
    main()
