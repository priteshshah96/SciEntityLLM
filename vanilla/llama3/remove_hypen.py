import json
import os

def process_json_files(directory):
    modified_files = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            modified = False
            
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            for item in data:
                if 'entity' in item and isinstance(item['entity'], str):
                    if item['entity'].startswith('- '):
                        item['entity'] = item['entity'][2:]
                        modified = True
                    elif item['entity'].startswith('-'):
                        item['entity'] = item['entity'][1:]
                        modified = True
            
            if modified:
                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=2)
                modified_files.append(filename)
    
    return modified_files

# Replace this with the actual path to your directory
directory = 'llama3_generated_entity_test'

modified_files = process_json_files(directory)

print("Files modified:")
for file in modified_files:
    print(file)

print(f"\nTotal files modified: {len(modified_files)}")