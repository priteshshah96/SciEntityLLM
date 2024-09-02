import os
import json
import torch
import time
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from contextlib import contextmanager

# Define the path for the output folder
generated_output_dir = './gemma2_generated_entities_test'
os.makedirs(generated_output_dir, exist_ok=True)
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

# Clear CUDA memory at the start of the script
clear_cuda_memory()

# Load the tokenizer and model from cache
model_id = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./cache")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir="./cache", device_map="auto", use_cache=False)
generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_text_with_options(prompt):
    with torch_gc():
        generated = generation_pipeline(prompt, max_length=4096, do_sample=false, num_return_sequences=1, truncation=True)[0]['generated_text']
    stop_token = "###"
    if stop_token in generated:
        generated = generated.split(stop_token)[0]
    return generated

def extract_generated_text(generated_text):
    split_text = generated_text.split('--- BEGIN NEW TEXT ---')
    if len(split_text) > 1:
        return split_text[1].strip().split('--- END NEW TEXT ---')[0].strip()
    return generated_text.strip()

def print_full_generated_text(generated_text):
    print("Full generated text:")
    print(generated_text)
    print("=" * 80)

def parse_entities(extracted_text):
    entities = []
    current_entity = None
    categories = ["Method", "Task", "Material", "Metric", "OtherScientificTerm"]
    default_category = "OtherScientificTerm"
    
    for line in extracted_text.split('\n'):
        line = line.strip()
        if line.startswith('*'):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"entity": line.strip('* ').strip(), "category": default_category}
        elif line.startswith('-') and current_entity:
            reasoning = line.strip('- ').strip()
            for category in categories:
                if category.lower() in reasoning.lower():
                    current_entity["category"] = category
                    break
    if current_entity:
        entities.append(current_entity)
    return entities

prompt_template = """
Extract the scientific entities from the following text by following these steps:

Identifying Key Entities: Identify the main scientific subjects and objects in each sentence, focusing on precise and technical nouns or noun phrases.
Domain-Specific Focus: Pay special attention to entities related to scientific and technical domains, including:

Task: The specific scientific or technical task being discussed (e.g., parsing, speech recognition)
Method: Techniques, algorithms, or approaches used (e.g., probabilistic parser, boosting method)
Metric: Measures used to evaluate performance (e.g., F-measure)
Material: Datasets, resources, or tools used in the research (e.g., Wall Street Journal treebank)
OtherScientificTerm: Any other relevant scientific concepts or terminology

Here are three examples that demonstrate these principles:
Example 1:
Text: "This paper introduces a system for categorizing unknown words. The system is based on a multi-component architecture where each component is responsible for identifying one class of unknown words. The focus of this paper is the components that identify names and spelling errors. Each component uses a decision tree architecture to combine multiple types of evidence about the unknown word. The system is evaluated using data from live closed captions - a genre replete with a wide variety of unknown words."

Scientific Entities Identified:

* categorizing unknown words
Category: Task

* multi-component architecture
Category: Method

* unknown words
Category: OtherScientificTerm

* names
Category: OtherScientificTerm

* spelling errors
Category: OtherScientificTerm

* decision tree architecture
Category: Method

* live closed captions
Category: Material

Example 2:
Text: "In some auction domains, there is uncertainty regarding the final availability of the goods being auctioned off. For example, a government may auction off spectrum from its public safety network, but it may need this spectrum back in times of emergency. In such a domain, standard combinatorial auctions perform poorly because they lead to violations of individual rationality (IR), even in expectation, and to very low efficiency. In this paper, we study the design of core-selecting payment rules for such domains. Surprisingly, we show that in this new domain, there does not exist a payment rule with is guaranteed to be ex-post core-selecting. However, we show that by designing rules that are "execution-contingent," i.e., by charging payments that are conditioned on the realization of the availability of the goods, we can reduce IR violations. We design two core-selecting rules that always satisfy IR in expectation. To study the performance of our rules we perform a computational Bayes-Nash equilibrium analysis. We show that, in equilibrium, our new rules have better incentives, higher efficiency, and a lower rate of ex-post IR violations than standard core-selecting rules."

Scientific Entities Identified:

* auction domains
Category: Task

* combinatorial auctions
Category: Method

* individual rationality (IR)
Category: OtherScientificTerm

* core-selecting payment rules
Category: Method

* execution-contingent rules
Category: Method

* IR violations
Category: Metric

* computational Bayes-Nash equilibrium analysis
Category: Method

* efficiency
Category: Metric

Example 3:
Text: "A " graphics for vision " approach is proposed to address the problem of reconstruction from a large and imperfect data set: reconstruction on demand by tensor voting, or ROD-TV. ROD-TV simultaneously delivers good efficiency and robust-ness, by adapting to a continuum of primitive connectivity, view dependence, and levels of detail (LOD). Locally inferred surface elements are robust to noise and better capture local shapes. By inferring per-vertex normals at sub-voxel precision on the fly, we can achieve interpolative shading. Since these missing details can be recovered at the current level of detail, our result is not upper bounded by the scanning resolution. By relaxing the mesh connectivity requirement, we extend ROD-TV and propose a simple but effective multiscale feature extraction algorithm. ROD-TV consists of a hierarchical data structure that encodes different levels of detail. The local reconstruction algorithm is tensor voting. It is applied on demand to the visible subset of data at a desired level of detail , by traversing the data hierarchy and collecting tensorial support in a neighborhood. We compare our approach and present encouraging results."

Scientific Entities Identified:

* "graphics for vision" approach
Category: Method

* reconstruction
Category: Task

* large and imperfect data set
Category: Material

* reconstruction
Category: Task

* tensor voting
Category: Method

* ROD-TV
Category: Method

* ROD-TV
Category: Method

* efficiency
Category: Metric

* robust-ness
Category: Metric

* primitive connectivity
Category: OtherScientificTerm

* view dependence
Category: OtherScientificTerm

* levels of detail (LOD)
Category: OtherScientificTerm

* Locally inferred surface elements
Category: OtherScientificTerm

* noise
Category: OtherScientificTerm

* local shapes
Category: OtherScientificTerm

* per-vertex normals
Category: OtherScientificTerm

* sub-voxel precision
Category: Metric

* interpolative shading
Category: Task

* scanning resolution
Category: OtherScientificTerm

* mesh connectivity requirement
Category: OtherScientificTerm

* ROD-TV
Category: Method

* multiscale feature extraction algorithm
Category: Method

* ROD-TV
Category: Method

* hierarchical data structure
Category: Method

* local reconstruction algorithm
Category: Method

* tensor voting
Category: Method

* traversing the data hierarchy
Category: Method

* collecting tensorial support
Category: Method


--- BEGIN NEW TEXT ---
Text: "{text}"
Scientific Entities Identified:
"""

# Set the path to the raw text files
raw_text_dir = "/home/shahprit/explanation/test"

# Load the list of processed files
try:
    with open('processed_files.json', 'r') as f:
        processed_files = set(json.load(f))
except FileNotFoundError:
    processed_files = set()

# Process all documents
files_not_matched = []
for filename in sorted(os.listdir(raw_text_dir)):
    if filename.endswith(".txt") and filename not in processed_files:
        file_path = os.path.join(raw_text_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            print(f"Processing file: {filename}")
            print("=" * 80)

            # Generate the prompt with the text
            prompt = prompt_template.format(text=text)

            # Generate text with the model
            generated_text = generate_text_with_options(prompt)

            # Print the full generated text
            print_full_generated_text(generated_text)

            # Extract the relevant part of the generated text
            extracted_text = extract_generated_text(generated_text)
            print("Extracted text:")
            print(extracted_text)
            print("=" * 80)

            # Count predicted and extracted entities
            predicted_count = extracted_text.count('*')
            extracted_count = 0
            lines = extracted_text.split('\n')
            for i in range(len(lines) - 1):
                if lines[i].strip().startswith('*') and lines[i+1].strip().startswith('-'):
                    extracted_count += 1
            
            print(f"Predicted entities: {predicted_count}")
            print(f"Extracted entities: {extracted_count}")
            
            # Parse the entities from the extracted text
            entities = parse_entities(extracted_text)
            
            print(f"Entities found: {len(entities)}")
            for entity in entities:
                print(f"- {entity['entity']}")
                print(f"  Category: {entity['category']}")
            print("=" * 80)
            
            # Check if counts match
            if predicted_count != len(entities) or extracted_count != len(entities):
                print(f"WARNING: Count mismatch in file {filename}")
                files_not_matched.append(filename)

            # Save the generated entities to a JSON file
            output_file_path = os.path.join(generated_output_dir, f"{os.path.splitext(filename)[0]}_generated_entities.json")
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                json.dump(entities, output_file, indent=2)

            print(f"Processed {filename}: Saved generated entities to {output_file_path}")
            print("=" * 80)

            # Mark file as processed
            processed_files.add(filename)
            with open('processed_files.json', 'w') as f:
                json.dump(list(processed_files), f)

        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")

        finally:
            # Clean up to free memory
            if 'generated_text' in locals():
                del generated_text
            if 'extracted_text' in locals():
                del extracted_text
            if 'entities' in locals():
                del entities
            clear_cuda_memory()
            time.sleep(2)  # Sleep to avoid overloading the model
    else:
        print(f"Skipping already processed file: {filename}")

print("Finished processing all documents.")

# Print list of files with count mismatches
if files_not_matched:
    print("Files with count mismatches:")
    for file in files_not_matched:
        print(file)
else:
    print("All files processed successfully with matching counts.")

script_dir = os.path.dirname(os.path.abspath(__file__))
files_not_matched_path = os.path.join(script_dir, 'files_not_matched.json')
with open(files_not_matched_path, 'w') as f:
    json.dump(files_not_matched, f, indent=2)

print(f"List of files with count mismatches saved to: {files_not_matched_path}")