#To-Do: write code to find errors in order of steps in the recipes
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

def json_len(f):
    #print(f)
    q = len(f)
    for i, j in enumerate(f):
        q += len(j['followup'])

    return q

def ground_truth(steps, n_annot, questions):
    # Create a list to store the followed questions
    followed_questions = []
    
    # Create a dictionary to store the step ids and whether they have an order error
    step_ids = {step['step_id']: 0 for step in n_annot['steps']}
    
    # Helper function to process a question and its followups
    def process_question(q, followups):
        nonlocal followed_questions
        
        # Check if the question has already been followed
        if q not in followed_questions:
            followed_questions.append(q)
        
        # Process follow-up questions
        for followup in followups:
            if followup not in followed_questions:
                followed_questions.append(followup)
    
    # Iterate through the questions and their followups
    for question in questions:
        q = question['q']
        followups = question.get('followup', [])
        process_question(q, followups)
    
    # Check for order errors and update step ids
    for step in steps['steps']:
        if step['has_errors'] and 'Order Error' in step.get('errors', []):
            step_ids[step['step_id']] = 1
    
    # Create the ground truth list
    gt = [step_ids[step['step_id']] for step in n_annot['steps']]
    
    return gt


def question_index(related_questions):
    question_to_index = {question['q']: i for i, question in enumerate(related_questions)}
    for i, question in enumerate(related_questions):
        for followup in question['followup']:
            question_to_index[followup] = i

    return question_to_index

def op_val(ans):
    if 'yes' in ans:
        return 0
    else:
        return 1

def data_file(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, sep=',', mode='a+')

def main():
    disable_torch_init()
    video_dir = '/home/ptg/ptg/rohith/resolution_360p/'
    questions_file = './error_prompts/order_error.json'
    gt_file = './step_annotations.json'
    normal_annot = './normal_videos.json'
    model_path = 'LanguageBind/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = 'cuda:1'
    output_file = './order_error.txt'
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
        video = os.path.join(video_dir, v)
        name = v.split("_")
        gt_name = name[0] + '_' + name[1]
        related_questions = qs[name[0] + "_x"]["questions"]
        gt = ground_truth(gt_f[gt_name], n_annot[name[0]+"_x"], related_questions)
        g_truth.append(gt)
        pred_op = [1] * len(gt)

        question_ind = question_index(related_questions)

        # Iterate over the related questions with progress tracking using tqdm
        for steps in tqdm(related_questions, desc=f"Processing questions for {v}", leave=False):
            inp1 = steps['q']
            pred = process_video(video, inp1, tokenizer, model, processor)
            pred = pred.lower()
            op = op_val(pred)
            all_pred2 = True
            for q_s in steps['followup']:
                inp2 = q_s
                pred2 = process_video(video, inp2, tokenizer, model, processor).lower()
                pred_op[question_ind[inp2]] = op_val(pred2)
                if op_val(pred2)!=0:
                    all_pred2 = False
            final_op = (0 and op) if all_pred2 else (1 and op)
            pred_op[question_ind[inp1]] = final_op
        
        predicted.append(pred_op)

    # Validate that predicted and g_truth are lists of lists
    assert all(isinstance(i, list) for i in predicted), "predicted is not a list of lists"
    assert all(isinstance(i, list) for i in g_truth), "g_truth is not a list of lists"

    predicted = flatten(predicted)
    g_truth = flatten(g_truth)
    print('predicted: ',predicted)
    print('ground_truth: ',g_truth)

    content = "Ground Truth: {g_truth} \n Predicted: {predicted}".format(
        g_truth = g_truth,
        predicted = predicted
    )

    with open(output_file, 'w') as file:
        file.write(content) 
           
if __name__ == '__main__':
    main()
