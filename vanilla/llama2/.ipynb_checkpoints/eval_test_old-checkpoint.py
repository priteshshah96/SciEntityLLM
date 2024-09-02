import json
import os
import re
from collections import Counter
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    parts = re.split(r'\s*(?:,|\sand\s|\splus\s)\s*', entity)
    return [part.strip() for part in parts if part.strip()]

def calculate_overlap(gold_entity, predicted_entity):
    gold_words = set(gold_entity.lower().split())
    predicted_words = set(predicted_entity.lower().split())
    
    overlap = len(gold_words.intersection(predicted_words))
    union = len(gold_words.union(predicted_words))

    return overlap / union if union > 0 else 0

def standardize_other_scientific_terms(entities):
    standardized = []
    for entity in entities:
        if entity['category'].lower() == 'other scientific term':
            # Standardize the spelling
            standardized_entity = entity['entity'].lower()
            # Remove any special characters and extra spaces
            standardized_entity = re.sub(r'[^a-z0-9\s]', '', standardized_entity)
            standardized_entity = ' '.join(standardized_entity.split())
            entity['entity'] = standardized_entity
        standardized.append(entity)
    return standardized

def extract_entities_and_categories(data):
    entities = []
    categories = []
    generic_terms = []

    if isinstance(data, dict) and 'entities' in data:
        for category, entity_list in data['entities'].items():
            for entity in entity_list:
                entities.append(entity['entity'])
                categories.append(entity['category'])
                if entity['category'] == 'Generic':
                    generic_terms.append(entity['entity'])
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and 'entity' in item and 'category' in item:
                entities.append(item['entity'])
                categories.append(item['category'])
                if item['category'] == 'Generic':
                    generic_terms.append(item['entity'])

    return entities, categories, generic_terms

def match_entities(gold_entities, gold_categories, predicted_entities, predicted_categories):
    exact_matches = []
    partial_matches = []
    split_matches = []
    category_matches = []
    gold_counter = Counter(zip(gold_entities, gold_categories))
    predicted_counter = Counter(zip(predicted_entities, predicted_categories))
    repeated_gold_entities = {}

    # First pass: Exact matches (including category)
    for (gold, gold_cat) in list(gold_counter.keys()):
        if gold_counter[(gold, gold_cat)] == 0:
            continue
        
        normalized_gold = normalize_entity(gold)
        for (predicted, pred_cat) in list(predicted_counter.keys()):
            if normalize_entity(predicted) == normalized_gold:
                match_count = min(gold_counter[(gold, gold_cat)], predicted_counter[(predicted, pred_cat)])
                exact_matches.extend([{"entity": gold, "category": gold_cat, "predicted_category": pred_cat}] * match_count)
                if gold_cat == pred_cat:
                    category_matches.extend([{"entity": gold, "category": gold_cat}] * match_count)
                gold_counter[(gold, gold_cat)] -= match_count
                predicted_counter[(predicted, pred_cat)] -= match_count
                if predicted_counter[(predicted, pred_cat)] == 0:
                    del predicted_counter[(predicted, pred_cat)]
                if gold_counter[(gold, gold_cat)] == 0:
                    del gold_counter[(gold, gold_cat)]
                break

    # Second pass: Split matches
    for (gold, gold_cat) in list(gold_counter.keys()):
        if gold_counter[(gold, gold_cat)] == 0:
            continue
        
        gold_parts = split_complex_entity(gold)
        abbreviation_match = re.search(r'\(([^)]+)\)', gold)
        if len(gold_parts) > 1 or abbreviation_match:
            matched_parts = []
            for part in gold_parts:
                best_match = None
                best_score = 0
                for (predicted, pred_cat) in list(predicted_counter.keys()):
                    score = calculate_overlap(normalize_entity(part), normalize_entity(predicted))
                    if score > best_score:
                        best_match = (predicted, pred_cat)
                        best_score = score
                if best_match and best_score >= 0.8:
                    matched_parts.append({"part": part, "matched": best_match[0], "predicted_category": best_match[1]})
                    predicted_counter[best_match] -= 1
                    if predicted_counter[best_match] == 0:
                        del predicted_counter[best_match]
            
            if abbreviation_match:
                abbreviation = abbreviation_match.group(1)
                for (predicted, pred_cat) in list(predicted_counter.keys()):
                    if predicted == abbreviation:
                        matched_parts.append({"part": abbreviation, "matched": abbreviation, "predicted_category": pred_cat})
                        predicted_counter[(predicted, pred_cat)] -= 1
                        if predicted_counter[(predicted, pred_cat)] == 0:
                            del predicted_counter[(predicted, pred_cat)]
                        break
            
            if len(matched_parts) >= 2 or (abbreviation_match and len(matched_parts) >= 1):
                split_matches.append({"gold": gold, "matched_parts": matched_parts, "category": gold_cat})
                if all(part["predicted_category"] == gold_cat for part in matched_parts):
                    category_matches.append({"entity": gold, "category": gold_cat})
                gold_counter[(gold, gold_cat)] -= 1
                if gold_counter[(gold, gold_cat)] == 0:
                    del gold_counter[(gold, gold_cat)]

    # Third pass: Partial matches
    for (gold, gold_cat) in list(gold_counter.keys()):
        if gold_counter[(gold, gold_cat)] == 0:
            continue
        
        normalized_gold = normalize_entity(gold)
        best_match = None
        best_score = 0
        for (predicted, pred_cat) in list(predicted_counter.keys()):
            score = calculate_overlap(normalized_gold, normalize_entity(predicted))
            if score >= 0.5 and score > best_score:
                best_match = (predicted, pred_cat)
                best_score = score
        
        if best_match:
            partial_matches.append({
                "gold": gold,
                "predicted": best_match[0],
                "score": best_score,
                "gold_category": gold_cat,
                "predicted_category": best_match[1]
            })
            if gold_cat == best_match[1]:
                category_matches.append({"entity": gold, "category": gold_cat})
            gold_counter[(gold, gold_cat)] -= 1
            predicted_counter[best_match] -= 1
            if predicted_counter[best_match] == 0:
                del predicted_counter[best_match]
            if gold_counter[(gold, gold_cat)] == 0:
                del gold_counter[(gold, gold_cat)]

    missing_entities = [{"entity": entity, "category": category} for (entity, category) in gold_counter.elements()]
    extra_entities = [{"entity": entity, "category": category} for (entity, category) in predicted_counter.elements()]

    # Handle repeated gold entities
    original_gold_counter = Counter(zip(gold_entities, gold_categories))
    for (gold, gold_cat), count in original_gold_counter.items():
        if count > 1:
            matched_count = sum(1 for match in exact_matches if match["entity"] == gold and match["category"] == gold_cat) + \
                            sum(1 for match in split_matches if match["gold"] == gold and match["category"] == gold_cat) + \
                            sum(1 for match in partial_matches if match["gold"] == gold and match["gold_category"] == gold_cat)
            if count > matched_count:
                repeated_gold_entities[f"{gold}|{gold_cat}"] = count - matched_count

    return exact_matches, partial_matches, split_matches, category_matches, missing_entities, extra_entities, repeated_gold_entities

def evaluate_categories(gold_categories, predicted_categories):
    # Filter out 'Generic' category
    gold_categories = [cat for cat in gold_categories if cat.lower() != 'generic']
    predicted_categories = [cat for cat in predicted_categories if cat.lower() != 'generic']

    category_counter = Counter(gold_categories)
    predicted_category_counter = Counter(predicted_categories)

    correct_categories = sum((category_counter & predicted_category_counter).values())
    total_predicted = sum(predicted_category_counter.values())
    total_gold = sum(category_counter.values())

    precision = correct_categories / total_predicted if total_predicted > 0 else 0.0
    recall = correct_categories / total_gold if total_gold > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1, correct_categories

def update_generated_entities_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    if isinstance(data, dict) and 'entities' in data:
        for category in data['entities']:
            data['entities'][category] = standardize_other_scientific_terms(data['entities'][category])
    elif isinstance(data, list):
        data = standardize_other_scientific_terms(data)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2)

def process_document(gold_file, gold_entities_dir, generation_dir, eval_dir):
    with open(os.path.join(gold_entities_dir, gold_file), 'r', encoding='utf-8') as file:
        gold_data = json.load(file)
    
    doc_key = gold_file.replace('.json', '')
    
    # Standardize gold entities
    if isinstance(gold_data, dict) and 'entities' in gold_data:
        for category in gold_data['entities']:
            gold_data['entities'][category] = standardize_other_scientific_terms(gold_data['entities'][category])
    elif isinstance(gold_data, list):
        gold_data = standardize_other_scientific_terms(gold_data)
    
    gold_entities, gold_categories, generic_entities = extract_entities_and_categories(gold_data)

    result_path = os.path.join(generation_dir, f"{doc_key}_generated_entities.json")

    if not os.path.exists(result_path):
        return None

    with open(result_path, 'r', encoding='utf-8') as result_file:
        predicted_data = json.load(result_file)
        
    # Standardize predicted entities
    if isinstance(predicted_data, dict) and 'entities' in predicted_data:
        for category in predicted_data['entities']:
            predicted_data['entities'][category] = standardize_other_scientific_terms(predicted_data['entities'][category])
    elif isinstance(predicted_data, list):
        predicted_data = standardize_other_scientific_terms(predicted_data)
        
    predicted_entities, predicted_categories, _ = extract_entities_and_categories(predicted_data)

    exact_matches, partial_matches, split_matches, category_matches, missing_entities, extra_entities, repeated_gold_entities = match_entities(gold_entities, gold_categories, predicted_entities, predicted_categories)

    # Generic matches
    generic_entity_matches = list((Counter(e["entity"] for e in extra_entities) & Counter(generic_entities)).elements())
    
    gold_category_counts = Counter(cat for cat in gold_categories if cat.lower() != 'generic')
    predicted_category_counts = Counter(cat for cat in predicted_categories if cat.lower() != 'generic')

    detailed_analysis = {
        "doc_key": doc_key,
        "gold_entities": [{"entity": entity, "category": category} for entity, category in zip(gold_entities, gold_categories)],
        "predicted_entities": [{"entity": entity, "category": category} for entity, category in zip(predicted_entities, predicted_categories)],
        "exact_matches": exact_matches,
        "partial_matches": partial_matches,
        "split_matches": split_matches,
        "category_matches": category_matches,
        "missing_entities": missing_entities,
        "extra_entities": extra_entities,
        "repeated_gold_entities": repeated_gold_entities,
        "generic_entities": generic_entities,
        "generic_entity_matches": generic_entity_matches,
        "gold_category_counts": dict(gold_category_counts),
        "predicted_category_counts": dict(predicted_category_counts),
        "counts": {
            "gold_entities": len(gold_entities),
            "predicted_entities": len(predicted_entities),
            "exact_matches": len(exact_matches),
            "partial_matches": len(partial_matches),
            "split_matches": len(split_matches),
            "category_matches": len(category_matches),
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
    category_precision, category_recall, category_f1, _ = evaluate_categories(gold_categories, predicted_categories)

    return {
        "doc_key": doc_key,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "category_precision": category_precision,
        "category_recall": category_recall,
        "category_f1": category_f1,
        "counts": detailed_analysis["counts"],
        "gold_category_counts": gold_category_counts,
        "predicted_category_counts": predicted_category_counts,
        "all_gold_category_counts": Counter(gold_categories),
        "all_predicted_category_counts": Counter(predicted_categories)
    }

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gold_entities_dir = os.path.join(base_dir, "gold_entities_test")
    generation_dir = os.path.join(base_dir, "llama2", "llama2_generated_entity_test")
    eval_dir = os.path.join(base_dir, "llama2", "llama2_eval_test")
    os.makedirs(eval_dir, exist_ok=True)

    logging.info(f"Base directory: {base_dir}")
    logging.info(f"Gold entities directory: {gold_entities_dir}")
    logging.info(f"Generation directory: {generation_dir}")
    logging.info(f"Evaluation directory: {eval_dir}")

    if not os.path.exists(gold_entities_dir):
        raise FileNotFoundError(f"The directory {gold_entities_dir} does not exist.")

    if not os.path.exists(generation_dir):
        raise FileNotFoundError(f"The directory {generation_dir} does not exist.")

    gold_files = sorted([f for f in os.listdir(gold_entities_dir) if f.endswith('.json')])
    logging.info(f"Number of gold files found: {len(gold_files)}")

    if len(gold_files) == 0:
        logging.warning("No gold files found. Please check the file naming convention and extension.")
        return

    # Update all generated entity files
    for gold_file in gold_files:
        doc_key = gold_file.replace('.json', '')
        result_path = os.path.join(generation_dir, f"{doc_key}_generated_entities.json")
        if os.path.exists(result_path):
            update_generated_entities_file(result_path)

    results = []
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_document, gold_file, gold_entities_dir, generation_dir, eval_dir): gold_file for gold_file in gold_files}
        for future in as_completed(future_to_file):
            gold_file = future_to_file[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    logging.info(f"Completed processing document: {result['doc_key']}")
                else:
                    logging.warning(f"No predicted entities found for document: {gold_file.replace('.json', '')}")
            except Exception as exc:
                logging.error(f"An error occurred while processing {gold_file}: {exc}")

    if not results:
        logging.error("No documents were successfully processed.")
        return

    overall_metrics = {
        "entity_metrics": {
            "precision": sum(r["precision"] for r in results) / len(results),
            "recall": sum(r["recall"] for r in results) / len(results),
            "f1_score": sum(r["f1"] for r in results) / len(results),
        },
        "category_metrics": {
            "precision": sum(r["category_precision"] for r in results) / len(results),
            "recall": sum(r["category_recall"] for r in results) / len(results),
            "f1_score": sum(r["category_f1"] for r in results) / len(results),
        },
        "counts": {
            "documents_processed": len(results),
            "total_gold_entities": sum(r["counts"]["gold_entities"] for r in results),
            "total_predicted_entities": sum(r["counts"]["predicted_entities"] for r in results),
            "total_exact_matches": sum(r["counts"]["exact_matches"] for r in results),
            "total_partial_matches": sum(r["counts"]["partial_matches"] for r in results),
            "total_split_matches": sum(r["counts"]["split_matches"] for r in results),
            "total_category_matches": sum(r["counts"]["category_matches"] for r in results),
            "total_missing_entities": sum(r["counts"]["missing_entities"] for r in results),
            "total_extra_entities": sum(r["counts"]["extra_entities"] for r in results),
            "total_repeated_gold_entities": sum(r["counts"]["repeated_gold_entities"] for r in results),
            "total_generic_entities": sum(r["counts"]["generic_entities"] for r in results),
            "total_generic_entity_matches": sum(r["counts"]["generic_entity_matches"] for r in results)
        },
        "gold_category_counts": Counter(),
        "predicted_category_counts": Counter(),
        "all_gold_category_counts": Counter(),
        "all_predicted_category_counts": Counter()
    }

    for r in results:
        overall_metrics["gold_category_counts"].update(r["gold_category_counts"])
        overall_metrics["predicted_category_counts"].update(r["predicted_category_counts"])
        overall_metrics["all_gold_category_counts"].update(r["all_gold_category_counts"])
        overall_metrics["all_predicted_category_counts"].update(r["all_predicted_category_counts"])

    overall_metrics["gold_category_counts"] = dict(overall_metrics["gold_category_counts"])
    overall_metrics["predicted_category_counts"] = dict(overall_metrics["predicted_category_counts"])
    overall_metrics["all_gold_category_counts"] = dict(overall_metrics["all_gold_category_counts"])
    overall_metrics["all_predicted_category_counts"] = dict(overall_metrics["all_predicted_category_counts"])

    logging.info("\nOverall Metrics:")
    logging.info(json.dumps(overall_metrics, indent=2))

    with open(os.path.join(eval_dir, "overall_metrics.json"), 'w', encoding='utf-8') as metrics_file:
        json.dump(overall_metrics, metrics_file, indent=2)

    logging.info("Evaluation complete.")

if __name__ == "__main__":
    main()