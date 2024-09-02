import json
import os
from collections import Counter

def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file: {file_path}")
        return None

def normalize_entity(entity):
    return ' '.join(entity.lower().split())

def calculate_overlap(generated_entity, gold_entity):
    generated_words = set(normalize_entity(generated_entity).split())
    gold_words = set(normalize_entity(gold_entity).split())
    intersection = len(generated_words.intersection(gold_words))
    return intersection / len(generated_words) if generated_words else 0

def compare_entities(gold_file, generated_file):
    gold_data = load_json(gold_file)
    generated_data = load_json(generated_file)

    if gold_data is None or generated_data is None:
        return None

    categories = ['Material', 'Method', 'Metric', 'Task', 'OtherScientificTerm']

    generic_entities = []
    gold_entities = []
    for category in categories:
        for entity in gold_data['entities'].get(category, []):
            entity_info = {
                'entity': normalize_entity(entity['entity']),
                'category': category
            }
            gold_entities.append(entity_info)

    for entity in gold_data['entities'].get('Generic', []):
        generic_entities.append({
            'entity': normalize_entity(entity['entity']),
            'category': 'Generic'
        })

    generated_entities = [
        {'entity': normalize_entity(item['entity']), 'category': item['category']}
        for item in generated_data
    ]

    gold_count = len(gold_entities)
    generated_count = len(generated_entities)
    generic_count = len(generic_entities)

    exact_matches = []
    exact_category_matches = []
    split_matches = []
    partial_matches = []

    # First pass: find exact matches
    for p_entity in generated_entities[:]:
        for g_entity in gold_entities[:]:
            if p_entity['entity'] == g_entity['entity']:
                exact_matches.append({
                    'gold': g_entity,
                    'predicted': p_entity
                })
                if p_entity['category'] == g_entity['category']:
                    exact_category_matches.append({
                        'gold': g_entity,
                        'predicted': p_entity
                    })
                generated_entities.remove(p_entity)
                gold_entities.remove(g_entity)
                break

    # Second pass: find split matches
    for g_entity in gold_entities[:]:
        matched_parts = []
        total_coverage = 0
        for p_entity in generated_entities[:]:
            overlap = calculate_overlap(p_entity['entity'], g_entity['entity'])
            if overlap > 0:
                matched_parts.append(p_entity)
                total_coverage += overlap

        if len(matched_parts) > 1 and total_coverage > 0.8:  # Consider it a split match if 80% or more is covered
            split_matches.append({
                'gold': g_entity,
                'predicted': matched_parts,
                'coverage': total_coverage
            })
            gold_entities.remove(g_entity)
            for part in matched_parts:
                generated_entities.remove(part)

    # Third pass: find partial matches
    for g_entity in gold_entities[:]:
        best_match = None
        best_overlap = 0
        for p_entity in generated_entities[:]:
            overlap = calculate_overlap(p_entity['entity'], g_entity['entity'])
            if overlap > 0.5 and overlap > best_overlap:
                best_match = p_entity
                best_overlap = overlap
        
        if best_match:
            partial_matches.append({
                'gold': g_entity,
                'predicted': best_match,
                'overlap': best_overlap
            })
            gold_entities.remove(g_entity)
            generated_entities.remove(best_match)

    extra_entities = generated_entities
    missing_entities = gold_entities

    all_gold_entities = [normalize_entity(e['entity']) for category in categories
                         for e in gold_data['entities'].get(category, [])]
    repeated_entities = {entity: count for entity, count in Counter(all_gold_entities).items() if count > 1}
    distinct_repeated_entities = list(repeated_entities.keys())

    return {
        'gold_entities': [{'entity': normalize_entity(e['entity']), 'category': e['category']} 
                          for category in categories
                          for e in gold_data['entities'].get(category, [])],
        'generated_entities': [{'entity': normalize_entity(e['entity']), 'category': e['category']} for e in generated_data],
        'generic_entities': generic_entities,
        'exact_matches': exact_matches,
        'exact_category_matches': exact_category_matches,
        'split_matches': split_matches,
        'partial_matches': partial_matches,
        'missing_entities': missing_entities,
        'extra_entities': extra_entities,
        'repeated_entities': repeated_entities,
        'distinct_repeated_entities': distinct_repeated_entities,
        'counts': {
            'gold_count': gold_count,
            'generated_count': generated_count,
            'generic_count': generic_count,
            'exact_matches': len(exact_matches),
            'exact_category_matches': len(exact_category_matches),
            'split_matches': len(split_matches),
            'partial_matches': len(partial_matches),
            'distinct_repeated_entities': len(distinct_repeated_entities),
            'repeated_entities': sum(repeated_entities.values())
        }
    }

def process_files(generated_dir, gold_dir, eval_dir):
    if not os.path.exists(generated_dir):
        print(f"Error: Generated entities directory not found: {generated_dir}")
        return

    if not os.path.exists(gold_dir):
        print(f"Error: Gold entities directory not found: {gold_dir}")
        return

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    overall_metrics = {
        'total_gold_entities': 0,
        'total_generated_entities': 0,
        'total_generic_entities': 0,
        'total_exact_matches': 0,
        'total_exact_category_matches': 0,
        'total_split_matches': 0,
        'total_partial_matches': 0,
        'total_distinct_repeated_entities': 0,
        'total_overall_repeated_entities': 0
    }

    for filename in os.listdir(generated_dir):
        if filename.endswith('_generated.json'):
            gold_filename = filename.replace('_generated.json', '.json')
            gold_file_path = os.path.join(gold_dir, gold_filename)
            if os.path.exists(gold_file_path):
                print(f"Processing: {filename}")
                try:
                    results = compare_entities(
                        gold_file_path,
                        os.path.join(generated_dir, filename)
                    )
                    if results:
                        save_results(results, os.path.join(eval_dir, filename))
                        overall_metrics['total_gold_entities'] += results['counts']['gold_count']
                        overall_metrics['total_generated_entities'] += results['counts']['generated_count']
                        overall_metrics['total_generic_entities'] += results['counts']['generic_count']
                        overall_metrics['total_exact_matches'] += results['counts']['exact_matches']
                        overall_metrics['total_exact_category_matches'] += results['counts']['exact_category_matches']
                        overall_metrics['total_split_matches'] += results['counts']['split_matches']
                        overall_metrics['total_partial_matches'] += results['counts']['partial_matches']
                        overall_metrics['total_distinct_repeated_entities'] += results['counts']['distinct_repeated_entities']
                        overall_metrics['total_overall_repeated_entities'] += results['counts']['repeated_entities']
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
            else:
                print(f"Warning: Corresponding gold file not found for {filename}")

    precision = overall_metrics['total_exact_category_matches'] / overall_metrics['total_generated_entities'] if overall_metrics['total_generated_entities'] > 0 else 0
    recall = overall_metrics['total_exact_category_matches'] / overall_metrics['total_gold_entities'] if overall_metrics['total_gold_entities'] > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    overall_metrics['precision'] = precision
    overall_metrics['recall'] = recall
    overall_metrics['f1_score'] = f1_score

    save_results(overall_metrics, os.path.join(eval_dir, 'overall_metrics.json'))

def save_results(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")

# Usage
script_dir = os.path.dirname(os.path.abspath(__file__))
generated_dir = os.path.join(script_dir, 'llama3_generated_entity_test')
gold_dir = os.path.join(script_dir, '..', 'gold_entities_test')
eval_dir = os.path.join(script_dir, 'eval')

print(f"Generated entities directory: {generated_dir}")
print(f"Gold entities directory: {gold_dir}")
print(f"Evaluation output directory: {eval_dir}")

process_files(generated_dir, gold_dir, eval_dir)