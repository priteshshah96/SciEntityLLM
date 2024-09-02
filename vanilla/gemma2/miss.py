import os
import json

def list_empty_json_files(folder_path):
    empty_files = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    
                if not data:
                    empty_files.append(filename)
            except json.JSONDecodeError:
                # If the file is not a valid JSON, we'll consider it empty
                empty_files.append(filename)
    
    return empty_files

# Replace this with the actual path to your folder
folder_path = './gemma2_generated_entity_test'

empty_files = list_empty_json_files(folder_path)

if empty_files:
    print("Empty JSON files:")
    for file in empty_files:
        print(file)
else:
    print("No empty JSON files found.")