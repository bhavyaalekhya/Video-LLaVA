import json

def ground_truth(video, qs, n_annot):
    g_t = []
    for v, info in video.items():
        gt = []
        name = v.split("_")[0]+'_x'
        n_steps_desc = []
        n_steps = n_annot[name]['steps']
        q = qs[name]
        for step in n_steps:
            n_steps_desc.append(step['description'])

        for step in info['steps']:
            if step['description'] in n_steps_desc:
                if step['has_errors'] == True:
                    gt.append(0)
                else:
                    gt.append(1)


        if len(n_steps_desc)!=len(gt):
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

    print(lists)

if __name__=='__main__':
    main()