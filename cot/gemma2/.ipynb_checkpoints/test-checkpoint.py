import os
import json
import torch
import gc
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import sys

# Define the paths for the output folders
generated_output_dir = './gemma2_generated_entity_test'
full_text_output_dir = './gemma2_full_generated_text'
os.makedirs(generated_output_dir, exist_ok=True)
os.makedirs(full_text_output_dir, exist_ok=True)

def clear_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

# Clear CUDA memory at the start of the script
clear_cuda_memory()

print("Initial GPU memory allocated:", torch.cuda.memory_allocated(0))
print("Initial GPU memory reserved:", torch.cuda.memory_reserved(0))

# Load the tokenizer and model
model_id = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./cache")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir="./cache", device_map="auto")

# Custom Dataset
class TextDataset(Dataset):
    def __init__(self, directory, processed_files):
        self.directory = directory
        self.files = [f for f in os.listdir(directory) if f.endswith('.txt') and f not in processed_files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.files[idx])
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return self.files[idx], text

# Prompt template
prompt_template = """
You are an expert in natural language processing and scientific text analysis. Your task is to extract and categorize scientific entities from the given text. Please follow these steps carefully:

1. Read the given text carefully.
2. Identify and list all relevant scientific entities found in the text. The number of entities may vary depending on the content.
3. For each entity, provide a category and reasoning.
4. Use only these categories: Task, Method, Metric, Material, OtherScientificTerm
5. Use the following format for each entity:
   * Entity Name
     Category: [Category]
     Reasoning: [Reasoning]

Here are the examples demonstrating this process:

Example 1:
Text: "This paper introduces a system for categorizing unknown words. The system is based on a multi-component architecture where each component is responsible for identifying one class of unknown words. The focus of this paper is the components that identify names and spelling errors. Each component uses a decision tree architecture to combine multiple types of evidence about the unknown word. The system is evaluated using data from live closed captions - a genre replete with a wide variety of unknown words."

Scientific Entities Identified:

* categorizing unknown words
Category: Task
Reasoning: This describes the main objective of the system, which is a specific scientific task.

* multi-component architecture
Category: Method
Reasoning: This describes the structure of the system, which is a specific approach to solving the task.

* unknown words
Category: OtherScientificTerm
Reasoning: This is the target of the categorization task, a key concept in the research.

* names
Category: OtherScientificTerm
Reasoning: This is one category of unknown words the system identifies, a specific scientific concept.

* spelling errors
Category: OtherScientificTerm
Reasoning: Another category of unknown words the system identifies, also a specific scientific concept.

* decision tree architecture
Category: Method
Reasoning: This describes the approach used by each component, a specific algorithmic technique.

* live closed captions
Category: Material
Reasoning: This describes the data used for evaluation, a specific dataset or resource.

Example 2:
Text: "In some auction domains, there is uncertainty regarding the final availability of the goods being auctioned off. For example, a government may auction off spectrum from its public safety network, but it may need this spectrum back in times of emergency. In such a domain, standard combinatorial auctions perform poorly because they lead to violations of individual rationality (IR), even in expectation, and to very low efficiency. In this paper, we study the design of core-selecting payment rules for such domains. Surprisingly, we show that in this new domain, there does not exist a payment rule with is guaranteed to be ex-post core-selecting. However, we show that by designing rules that are "execution-contingent," i.e., by charging payments that are conditioned on the realization of the availability of the goods, we can reduce IR violations. We design two core-selecting rules that always satisfy IR in expectation. To study the performance of our rules we perform a computational Bayes-Nash equilibrium analysis. We show that, in equilibrium, our new rules have better incentives, higher efficiency, and a lower rate of ex-post IR violations than standard core-selecting rules."

Scientific Entities Identified:

* auction domains
Category: Task
Reasoning: This describes the specific area of study in auctions, setting the context for the research task.

* combinatorial auctions
Category: Method
Reasoning: This is a specific type of auction mechanism, representing a methodological approach.

* individual rationality (IR)
Category: OtherScientificTerm
Reasoning: This is a concept in auction theory, fundamental to understanding the problem being addressed.

* core-selecting payment rules
Category: Method
Reasoning: This describes the specific type of rules being designed, representing a methodological approach.

* execution-contingent rules
Category: Method
Reasoning: This describes a specific approach to designing rules, representing a methodological innovation.

* IR violations
Category: Metric
Reasoning: This is used as a measure of performance for the auction rules.

* computational Bayes-Nash equilibrium analysis
Category: Method
Reasoning: This describes the approach used to study the performance of the rules.

* efficiency
Category: Metric
Reasoning: This is used as a measure of performance for the auction rules.

xample 3:
Text: "A " graphics for vision " approach is proposed to address the problem of reconstruction from a large and imperfect data set: reconstruction on demand by tensor voting, or ROD-TV. ROD-TV simultaneously delivers good efficiency and robust-ness, by adapting to a continuum of primitive connectivity, view dependence, and levels of detail (LOD). Locally inferred surface elements are robust to noise and better capture local shapes. By inferring per-vertex normals at sub-voxel precision on the fly, we can achieve interpolative shading. Since these missing details can be recovered at the current level of detail, our result is not upper bounded by the scanning resolution. By relaxing the mesh connectivity requirement, we extend ROD-TV and propose a simple but effective multiscale feature extraction algorithm. ROD-TV consists of a hierarchical data structure that encodes different levels of detail. The local reconstruction algorithm is tensor voting. It is applied on demand to the visible subset of data at a desired level of detail , by traversing the data hierarchy and collecting tensorial support in a neighborhood. We compare our approach and present encouraging results."

Scientific Entities Identified:

* "graphics for vision" approach
Category: Method
Reasoning: This describes the overall methodological approach used in the research.

* reconstruction
Category: Task
Reasoning: This is the main problem or task that the research aims to address.

* large and imperfect data set
Category: Material
Reasoning: This describes the type of data used in the research, which is a key resource or input.

* reconstruction
Category: Task
Reasoning: This is mentioned again, reinforcing that it's a key task in the research.

* tensor voting
Category: Method
Reasoning: This is a specific technique used for reconstruction in the research.

* ROD-TV
Category: Method
Reasoning: This is the name given to the specific method developed in this research, which stands for "reconstruction on demand by tensor voting".

* ROD-TV
Category: Method
Reasoning: Mentioned again, emphasizing its importance as the main method discussed.

* efficiency
Category: Metric
Reasoning: This is used to evaluate the performance of the ROD-TV method.

* robust-ness
Category: Metric
Reasoning: This is another metric used to evaluate the performance of the ROD-TV method.

* primitive connectivity
Category: OtherScientificTerm
Reasoning: This is a concept that the ROD-TV method adapts to, representing a specific aspect of the data or problem.

* view dependence
Category: OtherScientificTerm
Reasoning: This is another concept that the ROD-TV method adapts to, representing a specific aspect of the data or problem.

* levels of detail (LOD)
Category: OtherScientificTerm
Reasoning: This is a concept used in the method, representing different scales or resolutions of the data.

* Locally inferred surface elements
Category: OtherScientificTerm
Reasoning: This describes a specific aspect of the reconstruction process.

* noise
Category: OtherScientificTerm
Reasoning: This refers to unwanted variations in the data that the method is robust against.

* local shapes
Category: OtherScientificTerm
Reasoning: This refers to the geometric features that the method aims to capture accurately.

* per-vertex normals
Category: OtherScientificTerm
Reasoning: This is a specific type of data or feature used in the reconstruction process.

* sub-voxel precision
Category: Metric
Reasoning: This describes the level of accuracy achieved in the reconstruction process.

* interpolative shading
Category: Task
Reasoning: This is a specific task or capability achieved by the method.

* scanning resolution
Category: OtherScientificTerm
Reasoning: This refers to the resolution of the input data, which is a key concept in the research.

* mesh connectivity requirement
Category: OtherScientificTerm
Reasoning: This is a constraint or requirement that is relaxed in the extended version of the method.

* ROD-TV
Category: Method
Reasoning: Mentioned again in the context of extending the method.

* multiscale feature extraction algorithm
Category: Method
Reasoning: This is an additional algorithm proposed as an extension to ROD-TV.

* ROD-TV
Category: Method
Reasoning: Mentioned once more when describing its components.

* hierarchical data structure
Category: Method
Reasoning: This describes a key component of the ROD-TV method for encoding different levels of detail.

* local reconstruction algorithm
Category: Method
Reasoning: This refers to the specific algorithm used for local reconstruction in ROD-TV.

* tensor voting
Category: Method
Reasoning: Mentioned again as the specific local reconstruction algorithm used in ROD-TV.

* traversing the data hierarchy
Category: Method
Reasoning: This describes a specific process within the ROD-TV algorithm.

* collecting tensorial support
Category: Method
Reasoning: This is another specific process within the ROD-TV algorithm.

Now, apply this process to the following new text and identify all the scientific entities:

Text: "{text}"

Scientific Entities Identified:
"""

def generate_text(prompt, max_length=4096):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_entities(generated_text):
    entities_text = generated_text.split("Scientific Entities Identified:")[-1].strip()
    
    entities = []
    current_entity = {}
    
    for line in entities_text.split('\n'):
        line = line.strip()
        if line.startswith('*'):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"entity": line[1:].strip()}
        elif ':' in line:
            key, value = line.split(':', 1)
            current_entity[key.strip().lower()] = value.strip()
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

# Set the path to the raw text files
raw_text_dir = "/home/shahprit/cot/test"

# Load the list of processed files
try:
    with open('processed_files.json', 'r') as f:
        processed_files = set(json.load(f))
except FileNotFoundError:
    processed_files = set()

# Create dataset and dataloader
dataset = TextDataset(raw_text_dir, processed_files)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Process all documents
for filename, text in tqdm(dataloader, desc="Processing files"):
    filename = filename[0]  # Unpack from batch
    text = text[0]  # Unpack from batch
    
    try:
        print(f"\nProcessing file: {filename}")
        print("=" * 80)
        
        prompt = prompt_template.format(text=text)
        generated_text = generate_text(prompt)
        
        print("Generated text:")
        print(generated_text)
        print("=" * 80)
        
        entities = extract_entities(generated_text)
        
        print(f"Number of extracted entities: {len(entities)}")
        print("Extracted entities:")
        print(json.dumps(entities, indent=2))
        print("=" * 80)
        
        # Save the extracted entities
        output_file = os.path.join(generated_output_dir, f"{filename[:-4]}_generated.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(entities, f, ensure_ascii=False, indent=4)
        
        # Save the full generated text
        full_text_output_file = os.path.join(full_text_output_dir, f"{filename[:-4]}_full_generated.txt")
        with open(full_text_output_file, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        
        print(f"Saved outputs for {filename}")
        
        # Add to the processed files list
        processed_files.add(filename)
        with open('processed_files.json', 'w') as f:
            json.dump(list(processed_files), f)
        
    except Exception as e:
        print(f"Error processing file {filename}: {e}")

    # Clear CUDA memory after processing each file
    clear_cuda_memory()
    print(f"GPU memory after processing {filename}:")
    print("Allocated:", torch.cuda.memory_allocated(0))
    print("Reserved:", torch.cuda.memory_reserved(0))
    
    # Flush stdout to ensure real-time output
    sys.stdout.flush()

# Clear CUDA memory at the end of the script
clear_cuda_memory()

print("Final GPU memory allocated:", torch.cuda.memory_allocated(0))
print("Final GPU memory reserved:", torch.cuda.memory_reserved(0))

print("Processing complete.")