import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def get_valid_triplets(synth_data):
    i = 0 
    valid_triplets = set()
    for item in synth_data:
        for annotation in item['hoi_annotation']:
            subject_category = item["annotations"][annotation['subject_id']]['category_id']
            object_category = item["annotations"][annotation['object_id']]['category_id']
            role_category = annotation['category_id']
            if annotation['object_id'] >= 0 and object_category > 1:
                i += 1
                valid_triplets.add((subject_category, role_category, object_category))
    return i
    return valid_triplets


def read_json(files_path):
    json_data = []
    for file_path in files_path:
        with open(file_path, 'r') as f:
            file_data = json.load(f)
        json_data.extend(file_data)
    return json_data


def count_triplets(data, objects_filter=True):
    triplet_counter = Counter()

    for item in data:
        elements = item['annotations']
        for annotation in item['hoi_annotation']:
            subject_id = annotation['subject_id']
            object_id = annotation['object_id']

            subject_category = elements[subject_id]['category_id']
            role_category = annotation['category_id']
            object_category = elements[object_id]['category_id']
            if not objects_filter or (object_id >= 0 and object_category > 1):
                triplet = (subject_category, role_category, object_category)
                triplet_counter[triplet] += 1

    return triplet_counter


def plot_triplets(real_counter, synth_counter):
    # Set the style for paper-ready plots
    plt.style.use('seaborn-v0_8-paper')

    # Convert to inches for paper column width (3.5 inches is standard for single column)
    width_inches = 4.0
    height_inches = 2.5  # Adjust aspect ratio for better readability

    # Create figure with paper-friendly size
    plt.figure(figsize=(width_inches, height_inches), dpi=300)

    # Sort real data triplets by frequency
    sorted_real = sorted(real_counter.items(), key=lambda x: x[1], reverse=True)
    real_triplets, real_counts = zip(*sorted_real)

    # Get counts for synthetic data using same triplet order
    synth_counts = [synth_counter.get(triplet, 0) for triplet in real_triplets]

    # Plot with optimized parameters for paper
    bar_width = 1
    x = np.arange(len(real_counts))  # the label locations

    plt.bar(x - bar_width/2, real_counts, width=bar_width, color='#0066cc', label='Real', alpha=1.0, zorder=3)
    plt.bar(x + bar_width/2, synth_counts, width=bar_width, color='#cc0000', label='Synthetic', alpha=0.75, zorder=2)
    
    # Customize appearance for paper
    plt.xlabel('HOI categories', fontsize=8)
    plt.ylabel('N° instances', fontsize=8)
    plt.yscale('log')

    # Customize ticks
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.tick_params(axis='both', which='minor', labelsize=6)

    # Add grid but make it subtle
    # plt.grid(True, linestyle='--', alpha=0.3, zorder=1)

    # Customize legend
    plt.legend(fontsize=7, frameon=True, facecolor='white', edgecolor='none', bbox_to_anchor=(1.0, 1.0),
               loc='upper right')

    # Remove unnecessary whitespace
    plt.tight_layout()

    # Save with high DPI for paper quality
    plt.savefig('triplet_distributions.pdf', format='pdf', bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.savefig('triplet_distributions.png', format='png', bbox_inches='tight', pad_inches=0.02, dpi=300)

    plt.show()
    plt.close()

def main(real_path, synth_path):
    # Read and process real data
    real_data = read_json(real_path)
    real_counter = count_triplets(real_data)
    
    # Read and process synthetic data
    synth_data = read_json(synth_path)
    synth_counter = count_triplets(synth_data)
    
    # Plot distributions
    plot_triplets(real_counter, synth_counter)
    
    print(f"Total unique triplets (real): {len(real_counter)}")
    print(f"Total unique triplets (synthetic): {len(synth_counter)}")
    print("\nTop 10 most common triplets in real data:")
    for triplet, count in real_counter.most_common(10):
        print(f"{triplet}: {count} (synthetic: {synth_counter[triplet]})")

if __name__ == "__main__":
    from utils.utils import PROJECT_HOME
    real_annotations = f"{PROJECT_HOME}/data/hicodet/annotations/trainval_hico.json"
    synth_annotations = f"{PROJECT_HOME}/out/gen_fin2.json"
    main([real_annotations], [real_annotations, synth_annotations])