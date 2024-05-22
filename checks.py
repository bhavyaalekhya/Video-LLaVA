import json

def ground_truth(video):
    gt = []
    steps = video['steps']
    for step in steps:
        if step['has_errors']==True:
            gt.append(0)
        else:
            gt.append(1)
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