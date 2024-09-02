import json
import os
import re
from collections import Counter

def normalize_entity(entity):
    entity = re.sub(r'\s*([()[\]{}])\s*', r'\1', entity)
    entity = ' '.join(entity.split())
    return entity.lower()

def evaluate(gold, predicted):
    gold_counter = Counter(gold)
    predicted_counter = Counter(predicted)

    true_positives = sum((gold_counter & predicted_counter).values())
    false_positives = sum((predicted_counter - gold_counter).values())
    false_negatives = sum((gold_counter - predicted_counter).values())

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1

def split_complex_entity(entity):
    # Split on commas, 'and', or 'plus'
    parts = re.split(r'\s*(?:,|\sand\s|\splus\s)\s*', entity)
    return [part.strip() for part in parts if part.strip()]

def calculate_overlap(gold_entity, predicted_entity):
    gold_words = set(gold_entity.lower().split())
    predicted_words = set(predicted_entity.lower().split())
    
    overlap = len(gold_words.intersection(predicted_words))
    union = len(gold_words.union(predicted_words))

    return overlap / union if union > 0 else 0

def match_entities(gold_entities, predicted_entities):
    exact_matches = []
    partial_matches = []
    split_matches = []
    gold_counter = Counter(gold_entities)
    predicted_counter = Counter(predicted_entities)
    repeated_gold_entities = {}

    # First pass: Exact matches
    for gold in list(gold_counter.keys()):
        if gold_counter[gold] == 0:
            continue
        
        normalized_gold = normalize_entity(gold)
        for predicted in list(predicted_counter.keys()):
            if normalize_entity(predicted) == normalized_gold:
                match_count = min(gold_counter[gold], predicted_counter[predicted])
                exact_matches.extend([gold] * match_count)
                gold_counter[gold] -= match_count
                predicted_counter[predicted] -= match_count
                if predicted_counter[predicted] == 0:
                    del predicted_counter[predicted]
                if gold_counter[gold] == 0:
                    del gold_counter[gold]
                break

    # Second pass: Split matches
    for gold in list(gold_counter.keys()):
        if gold_counter[gold] == 0:
            continue
        
        gold_parts = split_complex_entity(gold)
        abbreviation_match = re.search(r'\(([^)]+)\)', gold)
        if len(gold_parts) > 1 or abbreviation_match:
            matched_parts = []
            for part in gold_parts:
                best_match = None
                best_score = 0
                for predicted in list(predicted_counter.keys()):
                    score = calculate_overlap(normalize_entity(part), normalize_entity(predicted))
                    if score > best_score:
                        best_match = predicted
                        best_score = score
                if best_match and best_score >= 0.8:
                    matched_parts.append((part, best_match))
                    predicted_counter[best_match] -= 1
                    if predicted_counter[best_match] == 0:
                        del predicted_counter[best_match]
            
            # Check for abbreviation match
            if abbreviation_match:
                abbreviation = abbreviation_match.group(1)
                if abbreviation in predicted_counter:
                    matched_parts.append((abbreviation, abbreviation))
                    predicted_counter[abbreviation] -= 1
                    if predicted_counter[abbreviation] == 0:
                        del predicted_counter[abbreviation]
            
            if len(matched_parts) >= 2 or (abbreviation_match and len(matched_parts) >= 1):
                split_matches.append((gold, matched_parts))
                gold_counter[gold] -= 1
                if gold_counter[gold] == 0:
                    del gold_counter[gold]

    # Third pass: Partial matches
    for gold in list(gold_counter.keys()):
        if gold_counter[gold] == 0:
            continue
        
        normalized_gold = normalize_entity(gold)
        best_match = None
        best_score = 0
        for predicted in list(predicted_counter.keys()):
            score = calculate_overlap(normalized_gold, normalize_entity(predicted))
            if score >= 0.5 and score > best_score:
                best_match = predicted
                best_score = score
        
        if best_match:
            partial_matches.append((gold, best_match, best_score))
            gold_counter[gold] -= 1
            predicted_counter[best_match] -= 1
            if predicted_counter[best_match] == 0:
                del predicted_counter[best_match]
            if gold_counter[gold] == 0:
                del gold_counter[gold]

    missing_entities = list(gold_counter.elements())
    extra_entities = list(predicted_counter.elements())

    # Handle repeated gold entities
    original_gold_counter = Counter(gold_entities)
    for gold, count in original_gold_counter.items():
        if count > 1:
            matched_count = sum(1 for match in exact_matches if match == gold) + \
                            sum(1 for match in split_matches if match[0] == gold) + \
                            sum(1 for match in partial_matches if match[0] == gold)
            if count > matched_count:
                repeated_gold_entities[gold] = count - matched_count

    return exact_matches, partial_matches, split_matches, missing_entities, extra_entities, repeated_gold_entities

def extract_entities(entities):
    if isinstance(entities, list):
        if all(isinstance(entity, str) for entity in entities):
            return entities
        elif all(isinstance(entity, dict) and 'entity' in entity for entity in entities):
            return [entity['entity'] for entity in entities]
    raise ValueError("Unexpected format for entities. Expected a list of strings or a list of dictionaries with 'entity' key.")

# Use absolute paths
base_dir = "/home/shahprit/vanilla"
gold_entities_dir = os.path.join(base_dir, "gold_entities")
generation_dir = os.path.join(base_dir, "gemma2", "gemma2a_generated_entity_train")
eval_dir = os.path.join(base_dir, "gemma2", "gemma2_eval_train")  # Corrected this line
os.makedirs(eval_dir, exist_ok=True)

# Print the paths for debugging
print(f"Gold entities directory: {gold_entities_dir}")
print(f"Generation directory: {generation_dir}")
print(f"Evaluation directory: {eval_dir}")

# Check if the gold_entities_dir exists
if not os.path.exists(gold_entities_dir):
    raise FileNotFoundError(f"The directory {gold_entities_dir} does not exist.")

gold_files = sorted([f for f in os.listdir(gold_entities_dir) if f.endswith('_entities.json')])

total_precision = total_recall = total_f1 = 0
total_exact_matches = total_partial_matches = total_split_matches = 0
total_extra_entities = total_missing_entities = total_predicted_entities = total_gold_entities = 0
total_generic_entities = total_repeated_gold_entities = 0
num_docs = 0
files_without_predicted_entities = []

for gold_file in gold_files:
    with open(os.path.join(gold_entities_dir, gold_file), 'r', encoding='utf-8') as file:
        gold_data = json.load(file)
    
    doc_key = gold_file.replace('_entities.json', '')
    gold_entities = extract_entities(gold_data['gold_entities'])
    generic_entities = extract_entities(gold_data.get('generic_terms', []))

    total_gold_entities += len(gold_entities)
    total_generic_entities += len(generic_entities)
    result_path = os.path.join(generation_dir, f"{doc_key}_generated_entities.json")

    if not os.path.exists(result_path):
        files_without_predicted_entities.append(doc_key)
        continue

    with open(result_path, 'r', encoding='utf-8') as result_file:
        predicted_data = json.load(result_file)
        predicted_entities = extract_entities(predicted_data)
    
    total_predicted_entities += len(predicted_entities)

    exact_matches, partial_matches, split_matches, missing_entities, extra_entities, repeated_gold_entities = match_entities(gold_entities, predicted_entities)

    total_exact_matches += len(exact_matches)
    total_partial_matches += len(partial_matches)
    total_split_matches += len(split_matches)
    total_missing_entities += len(missing_entities)
    total_extra_entities += len(extra_entities)
    total_repeated_gold_entities += sum(repeated_gold_entities.values())

    # Generic matches
    generic_entity_matches = list((Counter(extra_entities) & Counter(generic_entities)).elements())
    
    detailed_analysis = {
        "doc_key": doc_key,
        "gold_entities": gold_entities,
        "predicted_entities": predicted_entities,
        "exact_matches": exact_matches,
        "partial_matches": partial_matches,
        "split_matches": split_matches,
        "missing_entities": missing_entities,
        "extra_entities": extra_entities,
        "repeated_gold_entities": repeated_gold_entities,
        "generic_entities": generic_entities,
        "generic_entity_matches": generic_entity_matches,
        "counts": {
            "gold_entities": len(gold_entities),
            "predicted_entities": len(predicted_entities),
            "exact_matches": len(exact_matches),
            "partial_matches": len(partial_matches),
            "split_matches": len(split_matches),
            "missing_entities": len(missing_entities),
            "extra_entities": len(extra_entities),
            "repeated_gold_entities": sum(repeated_gold_entities.values()),
            "generic_entities": len(generic_entities),
            "generic_entity_matches": len(generic_entity_matches)
        }
    }

    with open(os.path.join(eval_dir, f"{doc_key}_analysis.json"), 'w', encoding='utf-8') as analysis_file:
        json.dump(detailed_analysis, analysis_file, indent=2)

    precision, recall, f1 = evaluate(gold_entities, predicted_entities)
    total_precision += precision
    total_recall += recall
    total_f1 += f1

    num_docs += 1
    print(f"Completed processing document: {doc_key}")

if num_docs > 0:
    avg_precision = total_precision / num_docs
    avg_recall = total_recall / num_docs
    avg_f1 = total_f1 / num_docs
else:
    avg_precision = avg_recall = avg_f1 = 0.0

overall_metrics = {
    "precision": avg_precision,
    "recall": avg_recall,
    "f1_score": avg_f1,
    "counts": {
        "documents_processed": num_docs,
        "total_gold_entities": total_gold_entities,
        "total_predicted_entities": total_predicted_entities,
        "total_exact_matches": total_exact_matches,
        "total_partial_matches": total_partial_matches,
        "total_split_matches": total_split_matches,
        "total_missing_entities": total_missing_entities,
        "total_extra_entities": total_extra_entities,
        "total_generic_entities": total_generic_entities,
        "total_repeated_gold_entities": total_repeated_gold_entities
    },
    "files_without_predicted_entities": files_without_predicted_entities
}

print("\nOverall Metrics:")
print(json.dumps(overall_metrics, indent=2))

with open(os.path.join(eval_dir, "overall_metrics.json"), 'w', encoding='utf-8') as metrics_file:
    json.dump(overall_metrics, metrics_file, indent=2)

print("Evaluation complete.")