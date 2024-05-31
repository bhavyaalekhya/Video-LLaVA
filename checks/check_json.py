import json

file = './error_prompts/measurement_error.json'

with open(file, 'r') as f:
    qs = json.load(f)

name = ['28']
related_key = name[0] + '_x'
data = qs[related_key]
related_questions = data['questions']
print(related_questions)