import json

def ground_truth(video):
    gt = []
    for v, info in video.items():
        g = []
        for step in info['steps']:
            if step['has_errors']==True:
                g.append(0)
            else:
                g.append(1)
        gt.append(g)
    return gt

def main():
    json_file = './step_annotations.json'
    with open(json_file, 'r') as f:
        cont = json.load(f)

    lists = ground_truth(cont)

    gt_flat = [label for sublist in lists for label in sublist]

    print(len(lists))
    print(len(gt_flat))

if __name__=='__main__':
    main()