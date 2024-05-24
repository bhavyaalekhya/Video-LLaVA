import json

import json

def ground_truth(video, qs, n_annot):
    g_t = []
    for v, info in video.items():
        gt = []
        name = v.split("_")[0] + '_x'
        n_steps_desc = []
        n_steps = n_annot[name]['steps']
        q = qs[name]
        
        # Collect normal step descriptions
        for step in n_steps:
            n_steps_desc.append(step['description'])

        # Get the set of common steps between normal steps and video steps
        common_steps = set(n_steps_desc).intersection(set([step['description'] for step in info['steps']]))

        # Initialize gt with default value (-1) for each common step
        gt = [-1] * len(common_steps)
        common_steps_list = list(common_steps)

        # Iterate over the video steps and match with normal steps descriptions
        for step in info['steps']:
            if step['description'] in common_steps:
                index = common_steps_list.index(step['description'])
                if step['has_errors']:
                    gt[index] = 0
                else:
                    gt[index] = 1

        g_t.append((v, len(n_steps_desc), len(gt)))

    return g_t

def main():
    json_file = './step_annotations.json'
    qs_file = './questions.json'
    n_annot = './normal_videos.json'
    with open(json_file, 'r') as f:
        cont = json.load(f)

    with open(qs_file, 'r') as file:
        qs = json.load(file)

    with open(n_annot, 'r') as f:
        n_steps = json.load(f)

    lists = ground_truth(cont, qs, n_steps)

    for i in lists:
        if i[1]!=i[2]:
            print(i)

if __name__ == '__main__':
    main()


def main():
    json_file = './step_annotations.json'
    qs_file = './questions.json'
    n_annot = './normal_videos.json'
    with open(json_file, 'r') as f:
        cont = json.load(f)

    with open(qs_file, 'r') as file:
        qs = json.load(file)

    with open(n_annot, 'r') as f:
        n_steps = json.load(f)

    lists = ground_truth(cont, qs, n_steps)

    print(lists)

if __name__ == '__main__':
    main()
