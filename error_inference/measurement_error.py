import torch
import os
import json
from tqdm import tqdm
import wandb
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

wandb.init(
    project="Task_Verification",
    entity="vsbhavyaalekhya"
)

def load_model(model_path, device, cache_dir, load_4bit=True, load_8bit=False):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    return tokenizer, model, processor

def process_video(video_path, question, tokenizer, model, processor):
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    video_tensor = video_processor(video_path, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)

    print(f"{roles[1]}: {question}")
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + question
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    return outputs

def flatten(l):
    return [label for sublist in l for label in sublist]

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
    gt = [1] * len(questions)

    for step in steps:
        if step['description'] in common_steps:
            index = common_steps.index(step['description'])
            if not step['has_errors']:
                gt[index] = 0

    return gt

def data_file(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, sep=',', mode='a+')

def dis(l):
    op = l[0]
    for i in range(1, len(l)):
        op = op and l[i]
    return op

def main():
    disable_torch_init()
    video_dir = '/data/rohith/captain_cook/videos/gopro/resolution_360p/'
    questions_file = './error_prompts/measurement_error.json'
    gt_file = './step_annotations.json'
    normal_annot = './normal_videos.json'
    model_path = 'LanguageBind/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = 'cuda'
    output_file = './error_inference/measurement_error.txt'
    load_4bit, load_8bit = True, False

    tokenizer, model, processor = load_model(model_path, device, cache_dir, load_4bit, load_8bit)

    with open(questions_file, 'r') as f:
        qs = json.load(f)

    with open(gt_file, 'r') as file:
        gt_f = json.load(file)

    with open(normal_annot, 'r') as f:
        n_annot = json.load(f)
    
    predicted = []
    g_truth = []

    for v in tqdm(os.listdir(video_dir), desc="Processing videos"):
        if v=='28_2_360p':
            video = os.path.join(video_dir, v)
            name = v.split("_")
            gt_name = name[0] + '_' + name[1]
            print(qs["28_x"])
            related_questions = qs[name[0] + "_x"]["questions"]
            pred_op = []
            gt = ground_truth(name[0], gt_f[gt_name], n_annot, related_questions)
            g_truth.append(gt)

            # Iterate over the related questions with progress tracking using tqdm
            for i in tqdm(range(len(related_questions)), desc=f"Processing questions for {v}", leave=False):
                inp = related_questions[i]['q']
                pred = process_video(video, inp, tokenizer, model, processor).lower()
                if 'yes' in pred:
                    if 'followup' in related_questions[i].keys():
                        preds = [0]
                        qs = related_questions[i]['followup']
                        for follow_up in qs:
                            pred2 = process_video(video, follow_up, tokenizer, model, processor).lower()
                            if 'yes' in pred2:
                                preds.append(0)
                            else:
                                preds.append(1)
                        p = dis(preds)
                        pred_op.append(p)
                    else:
                        pred_op.append(0)
                else:
                    pred_op.append(1)
        
        predicted.append(pred_op)

    predicted = flatten(predicted)
    g_truth = flatten(g_truth)
    print('predicted: ',predicted)
    print('ground_truth: ',g_truth)

    content = "Ground Truth: {g_truth} \n Predicted: {predicted}".format(
        g_truth = g_truth,
        predicted = predicted
    )

    #with open(output_file, 'w') as file:
    #    file.write(content) 
           
if __name__ == '__main__':
    main()
