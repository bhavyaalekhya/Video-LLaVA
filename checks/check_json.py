import json

file = './error_prompts/measurement_error.json'

with open(file, 'r') as f:
    qs = json.load(f)

l = qs['28_x']['questions']
print(len(l))