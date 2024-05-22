import json

def ground_truth(video, qs):
    gt = []
    for v, info in video.items():
        name = v.split("_")[0]+'_x'
        q = qs[name]
        gt.append((len(qs), len(info['steps'])))
    return gt

def main():
    json_file = './step_annotations.json'
    qs_file = './questions.json'
    with open(json_file, 'r') as f:
        cont = json.load(f)

    with open(qs_file, 'r') as file:
        qs = json.load(file)

    lists = ground_truth(cont, qs)


    print(len(lists))

if __name__=='__main__':
    main()