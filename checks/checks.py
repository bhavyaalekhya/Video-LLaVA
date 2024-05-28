import json

import json

def ground_truth(video, qs, n_annot):
    g_t = []
    for v, info in video.items():
        name = v.split("_")[0] + '_x'
        n_steps_desc = []
        n_steps = n_annot[name]['steps']
        q = qs[name]
        
        # Collect normal step descriptions
        for step in n_steps:
            n_steps_desc.append(step['description'])

        # Get the set of common steps between normal steps and video steps
        video_steps_desc = [step['description'] for step in info['steps']]
        common_steps = list(set(n_steps_desc).intersection(video_steps_desc))

        # Initialize gt with default value (-1) for each common step
        gt = [-1] * len(common_steps)
        
        # Populate gt based on video steps
        for step in info['steps']:
            if step['description'] in common_steps:
                index = common_steps.index(step['description'])
                if step['has_errors']:
                    gt[index] = 0
                else:
                    gt[index] = 1

        # Ensure gt has the same length as the common steps
        if len(gt) != len(common_steps):
            print(f"Length mismatch for video {v}: expected {len(common_steps)}, got {len(gt)}")
        g_t.append((v, len(n_steps_desc), len(common_steps)))

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

if __name__ == '__main__':
    main()
