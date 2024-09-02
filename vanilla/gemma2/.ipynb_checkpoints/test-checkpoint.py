import os
import json
import torch
import time
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from contextlib import contextmanager

# Define paths for output and cache
generated_output_dir = './gemma2_generated_entity_test'
cache_dir = "./cache"
os.makedirs(generated_output_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

@contextmanager
def torch_gc():
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def clear_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

# Clear CUDA memory at the start
clear_cuda_memory()

# Load the tokenizer and model from cache
model_id = "google/gemma-2-9b-it"  
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    cache_dir=cache_dir
)

generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

MAX_ENTITIES = 50

def generate_text_with_options(prompt):
    with torch_gc():
        generated = generation_pipeline(
            prompt, 
            max_length=2048,  
            num_return_sequences=1, 
            truncation=True
        )[0]['generated_text']
    return generated

prompt_template = """
Extract and list the unique scientific entities from the following text. Focus on technical terms, methods, metrics, tasks, materials, and other scientific concepts. Do not repeat entities. Categorize each entity using ONLY one of the following tags: Task, Method, Metric, Material, or Other-Scientific-Term.

- Scientific entity (Category)

Important:
- List each unique scientific entity only once.
- List each complete scientific entity on a new line, prefixed with a hyphen (-).
- Follow each entity with its category in parentheses.
- Use ONLY the categories: Task, Method, Metric, Material, or Other-Scientific-Term.
- Do not include any other text in your response.
- Limit your response to a maximum of 50 entities.

Text: "{text}"

Scientific entities:
"""

# def extract_entities(generated_text):
#     lines = generated_text.split('\n')
#     start_index = next((i for i, line in enumerate(lines) if "Scientific entities:" in line), -1)
    
#     if start_index == -1:
#         return [], 0  # No entities found
    
#     entities = []
#     valid_categories = {'Task', 'Method', 'Metric', 'Material', 'Other-Scientific-Term'}
#     total_entities = 0
    
#     for line in lines[start_index+1:]:
#         if line.strip().startswith('-'):
#             total_entities += 1
#             # Find the last occurrence of ' (' to split the entity and category
#             last_paren_index = line.rfind(' (')
#             if last_paren_index != -1:
#                 entity = line[1:last_paren_index].strip()
#                 category = line[last_paren_index+1:].strip('() ')
#                 # Remove the hyphen before the category if it exists
#                 if category.startswith('-'):
#                     category = category[1:]
#                 if category in valid_categories:
#                     entities.append({'entity': entity, 'category': category})
    
#     # Remove duplicates while preserving order
#     seen = set()
#     unique_entities = []
#     for entity in entities:
#         entity_key = (entity['entity'], entity['category'])
#         if entity_key not in seen:
#             seen.add(entity_key)
#             unique_entities.append(entity)
    
#     return unique_entities[:MAX_ENTITIES], total_entities


def extract_entities(generated_text):
    lines = generated_text.split('\n')
    start_index = next((i for i, line in enumerate(lines) if "Scientific entities:" in line), -1)
    
    if start_index == -1:
        return [], 0  # No entities found
    
    entities = []
    valid_categories = {'Task', 'Method', 'Metric', 'Material', 'Other-Scientific-Term'}
    total_entities = 0
    
    for line in lines[start_index+1:]:
        if line.strip().startswith('-'):
            total_entities += 1
            # Find the last occurrence of ' (' to split the entity and category
            last_paren_index = line.rfind(' (')
            if last_paren_index != -1:
                entity = line[1:last_paren_index].strip()
                category = line[last_paren_index+1:].strip('() ')
                # Remove the hyphen before the category if it exists
                if category.startswith('-'):
                    category = category[1:]
                if category in valid_categories:
                    entities.append({'entity': entity, 'category': category})
    
    return entities[:MAX_ENTITIES], total_entities

raw_text_dir = "/home/shahprit/vanilla/test"

# Track processed files
processed_files = set()
if os.path.exists('processed_files.json'):
    with open('processed_files.json', 'r') as f:
        processed_files = set(json.load(f))

files_to_process = [f for f in sorted(os.listdir(raw_text_dir)) if f.endswith(".txt") and f not in processed_files]

# List to store mismatched documents
files_to_remove = []
mismatched_documents = []

if not files_to_process:
    print("No new files to process. Exiting.")
else:
    print(f"Found {len(files_to_process)} new files to process.")
    
    for filename in files_to_process:
        file_path = os.path.join(raw_text_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            print(f"Processing file: {filename}")
            print("=" * 80)

            prompt = prompt_template.format(text=text)
            generated_text = generate_text_with_options(prompt)

            print("Full generated text:")
            print(generated_text)
            print("=" * 80)

            entities, total_entities = extract_entities(generated_text)

            print("Extracted entities:")
            for entity in entities:
                print(f"- {entity['entity']} ({entity['category']})")
            print("=" * 80)

            print(f"Entities found: {len(entities)}")
            print(f"Total entities in generated text: {total_entities}")

            if len(entities) != total_entities:
                mismatched_documents.append(filename)
                print(f"WARNING: Mismatch in entity count for {filename}")

            output_file_path = os.path.join(generated_output_dir, f"{os.path.splitext(filename)[0]}_generated_entities.json")
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                json.dump(entities, output_file, indent=2)

            print(f"Processed {filename}: Saved {len(entities)} entities to {output_file_path}")
            print("=" * 80)

            processed_files.add(filename)
            with open('processed_files.json', 'w') as f:
                json.dump(list(processed_files), f)

        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")

        finally:
            if 'generated_text' in locals():
                del generated_text
            if 'entities' in locals():
                del entities
            clear_cuda_memory()
            time.sleep(1)  # Avoid overloading the model

    print("Finished processing all new documents.")

    # After processing all files, update the files_to_remove list and print it
    if mismatched_documents:
        files_to_remove = mismatched_documents
        print("Documents with entity count mismatch:")
        print("files_to_remove = [")
        for doc in files_to_remove:
            print(f'    "{doc}",')
        print("]")
    else:
        print("No documents with entity count mismatch.")