import torch
import os
import json
from tqdm import tqdm
import wandb
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

def load_model():
    model_path = 'LanguageBind/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    return tokenizer, model, video_processor

def video_llava(video, inp, tokenizer, model, video_processor):
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)

    print(f"{roles[1]}: {inp}")
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
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

def accuracy(pred, gt):
    pred_pairs = [pair for sublist in pred for pair in sublist]
    gt_pairs = [pair for sublist in gt for pair in sublist]
    
    pred_flat = [label for pair in pred_pairs for label in pair]
    gt_flat = [label for pair in gt_pairs for label in pair]

    precision = precision_score(gt_flat, pred_flat)
    recall = recall_score(gt_flat, pred_flat)
    f1 = f1_score(gt_flat, pred_flat)
    accuracy = accuracy_score(gt_flat, pred_flat)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy
    }

def ground_truth(video, n_steps, related_questions):
    steps = video['steps']
    num_pairs = (len(related_questions) * (len(related_questions) - 1)) // 2
    gt_pairs = [(0,0)] * num_pairs
    
    pair_index = 0
    for i in range(len(related_questions)):
        for j in range(i + 1, len(related_questions)):
            first_step = steps[i]['step']
            second_step = steps[j]['step']
            if first_step in n_steps and second_step in n_steps:
                first_step_status = 1 if not steps[i]['has_errors'] else 0
                second_step_status = 1 if not steps[j]['has_errors'] else 0
                gt_pairs[pair_index] = (first_step_status, second_step_status)
                pair_index += 1

    return gt_pairs

def main():
    disable_torch_init()
    video_dir = '/data/rohith/captain_cook/videos/gopro/resolution_360p/'
    questions_file = './questions.json'
    gt_file = './step_annotations.json'
    normal_annot = './normal_videos.json'
    with open(questions_file, 'r') as f:
        qs = json.load(f)

    with open(gt_file, 'r') as file:
        gt_f = json.load(file)

    with open(normal_annot, 'r') as f:
        n_annot = json.load(f)
    
    tokenizer, model, video_processor = load_model()
    predicted = []
    g_truth = []
    wandb.watch(model, log='all')
    for v in tqdm(os.listdir(video_dir), desc="Processing videos"):
        video = os.path.join(video_dir, v)
        name = v.split("_")
        gt_name = name[0] + '_' + name[1]
        n_steps = n_annot[name[0]+'_x']['steps']
        n_steps_desc = []
        for step in n_steps:
            n_steps_desc.append(step['description'])
        gt = ground_truth(gt_f[gt_name], n_steps_desc, qs[name[0] + "_x"]["questions"])
        g_truth.append(gt)
        related_questions = qs[name[0] + "_x"]["questions"]
        pred_op = []

        # Generate predictions for all pairs of steps
        num_pairs = (len(related_questions) * (len(related_questions) - 1)) // 2
        print(f"Number of pairs for {v}: {num_pairs}")

        for i in range(len(related_questions)):
            for j in range(i + 1, len(related_questions)):
                q1 = related_questions[i]
                q2 = related_questions[j]
                
                pred1 = video_llava(video, q1, tokenizer, model, video_processor).lower()
                pred2 = video_llava(video, q2, tokenizer, model, video_processor).lower()
                
                pred1_op = 1 if 'yes' in pred1 else 0
                pred2_op = 1 if 'yes' in pred2 else 0
                
                pred_op.append((pred1_op, pred2_op))

        wandb.log({'gt': gt, 'pred': pred_op})
        predicted.append(pred_op)

    #metrics = accuracy(predicted, g_truth)

    #print(f"Accuracy: {metrics['accuracy']} \n F1: {metrics['f1_score']} \n Recall: {metrics['recall']} \n Precision: {metrics['precision']}")  

    with open('order_metrics.txt', 'a+') as file:
        content = '\n For pairwise step verification: Ground Truth: {g_truth} \nPredicted: {predicted}'.format(
            g_truth = g_truth,
            predicted = predicted
        )   
        file.write(content) 

if __name__ == '__main__':
    main()
