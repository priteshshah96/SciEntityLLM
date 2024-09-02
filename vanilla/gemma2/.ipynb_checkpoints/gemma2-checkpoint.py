import os
import json
import torch
import time
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from contextlib import contextmanager

# Define paths for output and cache
generated_output_dir = './gemma2a_generated_entity'
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
Extract and list the scientific entities from the following text. Focus on technical terms, methods, metrics, tasks, and concepts.

Important:
- Include repeated entities if they appear in different contexts or sentences.
- If an entity appears multiple times in the same context, list it only once for that context.
- List each complete scientific entity on a new line, prefixed with a hyphen (-).
- Do not include any other text in your response.

Text: "{text}"

Scientific entities:
"""

def extract_entities(generated_text):
    lines = generated_text.split('\n')
    start_index = next((i for i, line in enumerate(lines) if "Scientific entities:" in line), -1)
    
    if start_index == -1:
        return []  # No entities found
    
    entities = [line.strip('- "').strip() for line in lines[start_index+1:] if line.strip().startswith('-')]
    entities = [e for e in entities if not any(e.startswith(word) for word in ["Include", "If", "Do not"])]
    
    return entities

raw_text_dir = "/home/shahprit/vanilla/test"

# Track processed files
processed_files = set()
if os.path.exists('processed_files.json'):
    with open('processed_files.json', 'r') as f:
        processed_files = set(json.load(f))

files_to_process = [f for f in sorted(os.listdir(raw_text_dir)) if f.endswith(".txt") and f not in processed_files]

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

            entities = extract_entities(generated_text)[:MAX_ENTITIES]

            print("Extracted entities:")
            for entity in entities:
                print(f"- {entity}")
            print("=" * 80)

            print(f"Entities found: {len(entities)}")

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
            time.sleep(5)  # Avoid overloading the model

    print("Finished processing all new documents.")
