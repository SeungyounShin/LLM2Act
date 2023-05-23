import os,json

# /home/seungyoun/LLM2Act/demo_files/webshop_fever_react_only_success.json

new_json_file_path = './webshop_fever_react_only_success.json'
json_paths = [
    '/home/seungyoun/webshop/webshop/demo_files/webshop_demos_success_only.json',
    '/home/seungyoun/LLM2Act/FEVER/history/fever_llama_only_success.json',
]

new_list = []

for json_path in json_paths:
    with open(json_path, 'r') as f:
        data = json.load(f)

    for d in data:
        new_list.append(d)

# make json save code
# Save the new_list as a JSON file
with open(new_json_file_path, 'w') as f:
    json.dump(new_list, f)

# Print a message indicating the successful save
print(f"merged with {len(new_list)} demos")
