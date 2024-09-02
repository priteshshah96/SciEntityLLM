import json
import os
import re
from collections import Counter, defaultdict
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_entity(entity):
    entity = re.sub(r'\s*([()[\]{}])\s*', r'\1', entity)
    entity = ' '.join(entity.split())
    return entity.lower()

def normalize_category_name(category):
    normalized = re.sub(r'[^a-z0-9]', '', category.lower())
    if normalized in ['otherscientificterm', 'otherscientificterm']:
        return 'otherscientificterm'
    if normalized == '':
        logging.warning(f"Category '{category}' normalized to empty string")
    return normalized

def standardize_other_scientific_terms(entities):
    standardized = []
    for entity in entities:
        if normalize_category_name(entity['category']) == 'otherscientificterm':
            standardized_entity = entity['entity'].lower()
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
            normalized_category = normalize_category_name(category)
            for entity in entity_list:
                entities.append(entity['entity'])
                categories.append(normalized_category)
                if normalized_category == 'generic':
                    generic_terms.append(entity['entity'])
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and 'entity' in item and 'category' in item:
                entities.append(item['entity'])
                normalized_category = normalize_category_name(item['category'])
                categories.append(normalized_category)
                if normalized_category == 'generic':
                    generic_terms.append(item['entity'])

    return entities, categories, generic_terms

def calculate_overlap(gold_entity, predicted_entity):
    gold_words = set(gold_entity.lower().split())
    predicted_words = set(predicted_entity.lower().split())
    
    overlap = len(gold_words.intersection(predicted_words))
    union = len(gold_words.union(predicted_words))

    return overlap / union if union > 0 else 0

def split_complex_entity(entity):
    parts = re.split(r'\s*(?:,|\sand\s|\splus\s)\s*', entity)
    return [part.strip() for part in parts if part.strip()]

def match_entities(gold_entities, gold_categories, predicted_entities, predicted_categories, generic_entities):
    exact_matches = []
    exact_matches_with_category = []
    partial_matches = []
    split_matches = []
    generic_matches = []
    gold_counter = Counter(zip(gold_entities, gold_categories))
    predicted_counter = Counter(zip(predicted_entities, predicted_categories))

    # First, find all exact matches without considering category
    for (gold, gold_cat) in list(gold_counter.keys()):
        normalized_gold = normalize_entity(gold)
        for (predicted, pred_cat) in list(predicted_counter.keys()):
            if normalize_entity(predicted) == normalized_gold:
                match_count = min(gold_counter[(gold, gold_cat)], predicted_counter[(predicted, pred_cat)])
                if normalize_category_name(gold_cat) == normalize_category_name(pred_cat):
                    exact_matches_with_category.extend([{"gold": gold, "predicted": predicted, "category": gold_cat}] * match_count)
                else:
                    exact_matches.extend([{"gold": gold, "predicted": predicted, "gold_category": gold_cat, "predicted_category": pred_cat}] * match_count)
                gold_counter[(gold, gold_cat)] -= match_count
                predicted_counter[(predicted, pred_cat)] -= match_count
                if predicted_counter[(predicted, pred_cat)] == 0:
                    del predicted_counter[(predicted, pred_cat)]
                if gold_counter[(gold, gold_cat)] == 0:
                    del gold_counter[(gold, gold_cat)]
                break

    # Split matches
    for (gold, gold_cat) in list(gold_counter.keys()):
        gold_parts = split_complex_entity(gold)
        if len(gold_parts) > 1:
            matched_parts = []
            for part in gold_parts:
                for (predicted, pred_cat) in list(predicted_counter.keys()):
                    if normalize_entity(part) == normalize_entity(predicted):
                        matched_parts.append({"part": part, "matched": predicted, "predicted_category": pred_cat})
                        predicted_counter[(predicted, pred_cat)] -= 1
                        if predicted_counter[(predicted, pred_cat)] == 0:
                            del predicted_counter[(predicted, pred_cat)]
                        break
            if len(matched_parts) >= 2:
                split_matches.append({
                    "gold": gold,
                    "gold_category": gold_cat,
                    "matched_parts": matched_parts,
                    "num_predicted_entities": len(matched_parts)
                })
                gold_counter[(gold, gold_cat)] -= 1
                if gold_counter[(gold, gold_cat)] == 0:
                    del gold_counter[(gold, gold_cat)]

    # Partial matches
    for (gold, gold_cat) in list(gold_counter.keys()):
        normalized_gold = normalize_entity(gold)
        best_match = None
        best_score = 0
        for (predicted, pred_cat) in list(predicted_counter.keys()):
            score = calculate_overlap(normalized_gold, normalize_entity(predicted))
            if score > best_score:
                best_match = (predicted, pred_cat)
                best_score = score
        
        if best_match and best_score >= 0.5:
            partial_matches.append({
                "gold": gold,
                "predicted": best_match[0],
                "score": best_score,
                "gold_category": gold_cat,
                "predicted_category": best_match[1]
            })
            gold_counter[(gold, gold_cat)] -= 1
            predicted_counter[best_match] -= 1
            if predicted_counter[best_match] == 0:
                del predicted_counter[best_match]
            if gold_counter[(gold, gold_cat)] == 0:
                del gold_counter[(gold, gold_cat)]

    # Generic matches
    for (predicted, pred_cat) in list(predicted_counter.keys()):
        if normalize_category_name(pred_cat) == 'generic' and predicted in generic_entities:
            generic_matches.append({"entity": predicted, "category": pred_cat})
            predicted_counter[(predicted, pred_cat)] -= 1
            if predicted_counter[(predicted, pred_cat)] == 0:
                del predicted_counter[(predicted, pred_cat)]

    missing_entities = [{"entity": entity, "category": category} for (entity, category) in gold_counter.elements()]
    extra_entities = [{"entity": entity, "category": category} for (entity, category) in predicted_counter.elements()]

    return exact_matches, exact_matches_with_category, partial_matches, split_matches, generic_matches, missing_entities, extra_entities

def analyze_repeated_entities(gold_entities, gold_categories, exact_matches, exact_matches_with_category, partial_matches, split_matches):
    repeated_entities = defaultdict(lambda: {"count": 0, "exact": 0, "exact_with_category": 0, "partial": 0, "split": 0})
    
    # Count occurrences in gold standard
    for entity, category in zip(gold_entities, gold_categories):
        key = f"{entity}|{category}"
        repeated_entities[key]["count"] += 1
    
    # Count matches
    for match in exact_matches:
        key = f"{match['gold']}|{match['gold_category']}"
        repeated_entities[key]["exact"] += 1
    
    for match in exact_matches_with_category:
        key = f"{match['gold']}|{match['category']}"
        repeated_entities[key]["exact_with_category"] += 1
    
    for match in partial_matches:
        key = f"{match['gold']}|{match['gold_category']}"
        repeated_entities[key]["partial"] += 1
    
    for match in split_matches:
        key = f"{match['gold']}|{match['gold_category']}"
        repeated_entities[key]["split"] += 1
    
    # Filter out non-repeated entities
    repeated_entities = {k: v for k, v in repeated_entities.items() if v["count"] > 1}
    
    return repeated_entities

def evaluate_metrics(exact_matches_with_category, total_gold, total_predicted):
    correct = len(exact_matches_with_category)
    
    precision = correct / total_predicted if total_predicted > 0 else 0
    recall = correct / total_gold if total_gold > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1, correct

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

    # Identify entities with empty categories
    empty_category_entities = []
    for entity, category in zip(predicted_entities, predicted_categories):
        if normalize_category_name(category) == '':
            empty_category_entities.append({"entity": entity, "original_category": category})

    # Remove generic entities from gold_entities and gold_categories
    non_generic_gold = [(entity, category) for entity, category in zip(gold_entities, gold_categories) if normalize_category_name(category) != 'generic']
    gold_entities, gold_categories = zip(*non_generic_gold) if non_generic_gold else ([], [])

    exact_matches, exact_matches_with_category, partial_matches, split_matches, generic_matches, missing_entities, extra_entities = match_entities(gold_entities, gold_categories, predicted_entities, predicted_categories, generic_entities)

    repeated_entities = analyze_repeated_entities(gold_entities, gold_categories, exact_matches, exact_matches_with_category, partial_matches, split_matches)

    precision, recall, f1, correct = evaluate_metrics(
        exact_matches_with_category,
        len(gold_entities),
        len(predicted_entities)
    )

    gold_category_counts = Counter(normalize_category_name(cat) for cat in gold_categories if normalize_category_name(cat) != 'generic')
    predicted_category_counts = Counter(normalize_category_name(cat) for cat in predicted_categories if normalize_category_name(cat) != 'generic')

    # Category-based counts
    category_based_counts = defaultdict(lambda: {"gold": 0, "predicted": 0, "exact_matches": 0, "exact_matches_with_category": 0, "partial_matches": 0, "split_matches": 0})
    for cat in set(gold_category_counts.keys()) | set(predicted_category_counts.keys()):
        normalized_cat = normalize_category_name(cat)
        category_based_counts[normalized_cat]["gold"] += gold_category_counts[cat]
        category_based_counts[normalized_cat]["predicted"] += predicted_category_counts[cat]
        category_based_counts[normalized_cat]["exact_matches"] += sum(1 for match in exact_matches if normalize_category_name(match["gold_category"]) == normalized_cat)
        category_based_counts[normalized_cat]["exact_matches_with_category"] += sum(1 for match in exact_matches_with_category if normalize_category_name(match["category"]) == normalized_cat)
        category_based_counts[normalized_cat]["partial_matches"] += sum(1 for match in partial_matches if normalize_category_name(match["gold_category"]) == normalized_cat)
        category_based_counts[normalized_cat]["split_matches"] += sum(1 for match in split_matches if normalize_category_name(match["gold_category"]) == normalized_cat)

    detailed_analysis = {
        "doc_key": doc_key,
        "gold_entities": [{"entity": entity, "category": category} for entity, category in zip(gold_entities, gold_categories)],
        "predicted_entities": [{"entity": entity, "category": category} for entity, category in zip(predicted_entities, predicted_categories)],
        "exact_matches": exact_matches,
        "exact_matches_with_category": exact_matches_with_category,
        "partial_matches": partial_matches,
        "split_matches": split_matches,
        "generic_matches": generic_matches,
        "missing_entities": missing_entities,
        "extra_entities": extra_entities,
        "repeated_entities": repeated_entities,
        "repeated_entity_summary": {
            "total_distinct_repeated_entities": len(repeated_entities),
            "total_repeated_entities": sum(v["count"] for v in repeated_entities.values())
        },
        "gold_category_counts": dict(gold_category_counts),
        "predicted_category_counts": dict(predicted_category_counts),
        "category_based_counts": dict(category_based_counts),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "counts": {
            "gold_entities": len(gold_entities),
            "predicted_entities": len(predicted_entities),
            "exact_matches": len(exact_matches),
            "exact_matches_with_category": len(exact_matches_with_category),
            "partial_matches": len(partial_matches),
            "split_matches": len(split_matches),
            "generic_matches": len(generic_matches),
            "missing_entities": len(missing_entities),
            "extra_entities": len(extra_entities),
            "generic_entities": len(generic_entities)
        },
        "empty_category_entities": empty_category_entities
    }

    with open(os.path.join(eval_dir, f"{doc_key}_analysis.json"), 'w', encoding='utf-8') as analysis_file:
        json.dump(detailed_analysis, analysis_file, indent=2)

    return detailed_analysis

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gold_entities_dir = os.path.join(base_dir, "gold_entities_test")
    generation_dir = os.path.join(base_dir, "gemma2", "gemma2_generated_entities_test")
    eval_dir = os.path.join(base_dir, "gemma2", "gemma2_eval_test")
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
    documents_with_empty_categories = []

    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_document, gold_file, gold_entities_dir, generation_dir, eval_dir): gold_file for gold_file in gold_files}
        for future in as_completed(future_to_file):
            gold_file = future_to_file[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    if result['empty_category_entities']:
                        documents_with_empty_categories.append({
                            "doc_key": result['doc_key'],
                            "empty_category_entities": result['empty_category_entities']
                        })
                    logging.info(f"Completed processing document: {result['doc_key']}")
                else:
                    logging.warning(f"No predicted entities found for document: {gold_file.replace('.json', '')}")
            except Exception as exc:
                logging.error(f"An error occurred while processing {gold_file}: {exc}")

    if not results:
        logging.error("No documents were successfully processed.")
        return

    # Calculate overall metrics
    total_gold = sum(r['counts']['gold_entities'] for r in results)
    total_predicted = sum(r['counts']['predicted_entities'] for r in results)
    total_correct = sum(r['counts']['exact_matches_with_category'] for r in results)

    overall_precision = total_correct / total_predicted if total_predicted > 0 else 0
    overall_recall = total_correct / total_gold if total_gold > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if overall_precision + overall_recall > 0 else 0

    # Aggregate counts
    total_counts = {
        "gold_entities": sum(r['counts']['gold_entities'] for r in results),
        "predicted_entities": sum(r['counts']['predicted_entities'] for r in results),
        "exact_matches": sum(r['counts']['exact_matches'] for r in results),
        "exact_matches_with_category": sum(r['counts']['exact_matches_with_category'] for r in results),
        "partial_matches": sum(r['counts']['partial_matches'] for r in results),
        "split_matches": sum(r['counts']['split_matches'] for r in results),
        "generic_matches": sum(r['counts']['generic_matches'] for r in results),
        "missing_entities": sum(r['counts']['missing_entities'] for r in results),
        "extra_entities": sum(r['counts']['extra_entities'] for r in results),
        "generic_entities": sum(r['counts']['generic_entities'] for r in results)
    }

    # Aggregate category-based counts
    category_based_counts = defaultdict(lambda: {"gold": 0, "predicted": 0, "exact_matches": 0, "exact_matches_with_category": 0, "partial_matches": 0, "split_matches": 0})
    for r in results:
        for category, counts in r['category_based_counts'].items():
            for key, value in counts.items():
                category_based_counts[category][key] += value

    # Aggregate repeated entity summary
    total_distinct_repeated_entities = sum(len(r['repeated_entities']) for r in results)
    total_repeated_entities = sum(sum(v["count"] for v in r['repeated_entities'].values()) for r in results)

    # Generate overall metrics report
    overall_metrics = {
        "total_documents": len(results),
        "entity_metrics": {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1_score": overall_f1
        },
        "total_counts": total_counts,
        "category_based_counts": dict(category_based_counts),
        "repeated_entity_summary": {
            "total_distinct_repeated_entities": total_distinct_repeated_entities,
            "total_repeated_entities": total_repeated_entities
        }
    }

    # Save overall metrics
    with open(os.path.join(eval_dir, "overall_metrics.json"), 'w', encoding='utf-8') as metrics_file:
        json.dump(overall_metrics, metrics_file, indent=2)

    # Log documents with empty categories
    if documents_with_empty_categories:
        logging.warning("Documents with empty categories:")
        for doc in documents_with_empty_categories:
            logging.warning(f"Document: {doc['doc_key']}")
            for entity in doc['empty_category_entities']:
                logging.warning(f"  Entity: {entity['entity']}, Original category: {entity['original_category']}")

    # Save empty category documents information
    with open(os.path.join(eval_dir, "empty_category_documents.json"), 'w', encoding='utf-8') as f:
        json.dump(documents_with_empty_categories, f, indent=2)

    logging.info("Evaluation complete. Overall metrics:")
    logging.info(json.dumps(overall_metrics, indent=2))

if __name__ == "__main__":
    main()