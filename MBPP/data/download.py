from datasets import load_dataset
import json 

dataset_full = load_dataset("mbpp")

trainset = dataset_full['train']
testset = dataset_full['test']
promptset = dataset_full['prompt']

train_list= list()
for data in trainset:
    train_list.append(data)

test_list = list()
for data in trainset:
    test_list.append(data)

prompt_list = list()
for data in promptset:
    prompt_list.append(data)

print(f'train : {len(train_list)} test : {len(test_list)} prompt : {len(prompt_list)}')

# save as json 
with open('trainset.json', 'w') as train_file:
    json.dump(train_list, train_file)

# Save testset as JSON
with open('testset.json', 'w') as test_file:
    json.dump(test_list, test_file)

# Save promptset as JSON
with open('promptset.json', 'w') as prompt_file:
    json.dump(prompt_list, prompt_file)