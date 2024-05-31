import json

file = './error_prompts/measurement_error.json'

with open(file, 'r') as f:
    qs = json.load(f)

print(qs)