# This file can be ignored. Those are simple scripts used internally for plotting
from typing import Iterable
def create_images_plots(subfolders: Iterable[str], source_folder, out_folder="out_plots"):
    import os
    import matplotlib.pyplot as plt
    from PIL import Image
    image_dict = {}
    os.makedirs(os.path.join(os.path.dirname(subfolders[0]), out_folder), exist_ok=True)
    for folder in subfolders:
        for filename in os.listdir(folder):
            if filename.endswith('.jpg'):  # You can adjust for other formats if needed
                img_path = os.path.join(folder, filename)
                if filename not in image_dict:
                    image_dict[filename] = []
                image_dict[filename].append(img_path)
    sorted_image_names = sorted(image_dict.keys())
    last_id = None
    num = 0
    id_count = {}
    for image_name in sorted_image_names:
        image_id = image_name.rsplit('_', 1)[0]
        if image_id not in id_count:
            id_count[image_id] = 0
        paths = image_dict[image_name]
        num_images = 0
        for path in paths:
            if os.path.exists(path):
                num_images +=1
        if num_images == len(subfolders):
            id_count[image_id] += 1
    for image_name in sorted_image_names:
        paths = image_dict[image_name]
        image_id = image_name.rsplit('_', 1)[0]
        if False and id_count[image_id] < 3:
            continue  # Skip to the next image if not enough
        if last_id == image_id:
            num += 1
        else:
            last_id = image_id
            num = 0
        images = [Image.open(os.path.join(source_folder, image_id + f"_{num}.jpg"))]
        folder_names = ["Source"]
        for path in paths:
            folder_names.append(os.path.basename(os.path.dirname(path)))
            images.append(Image.open(path))
        num_images = len(images)
        if num_images != len(subfolders)+1:
            print(f"No all images found for {image_name}.")
            continue
        fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
        for ax, img, folder_name in zip(axes, images, folder_names):
            ax.imshow(img)
            ax.set_title(folder_name)
            ax.axis('off')  # Hide axes
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.05)
        output_image_path = os.path.join(os.path.dirname(subfolders[0]), out_folder, image_name)
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Created plot saved as: {output_image_path}")
# create_images_plots(["train_data_gen_vcoco", "train_data_gen_vcoco2"], "../data/vcoco_formatted/source_obj", "out_vcoco_plots")


def filter_im(json_path="gen_filter.json", images_folder=".", output_folder="filter"):
    import os
    import shutil
    import json
    import glob
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    with open(json_path, 'r') as f:
        content = f.read()
        data = json.loads(content)
    valid_filenames = {item['file_name'] for item in data}
    image_files = glob.glob(os.path.join(images_folder, '*.jpg'))
    moved_count = 0
    skipped_count = 0
    for image_path in image_files:
        filename = os.path.basename(image_path)
        if filename in valid_filenames:
            dest_path = os.path.join(output_folder, filename)
            try:
                shutil.move(image_path, dest_path)
                moved_count += 1
                print(f"Moved: {filename}")
            except Exception as e:
                print(f"Error moving {filename}: {str(e)}")
        else:
            skipped_count += 1
    return moved_count, skipped_count


def find_complete_bipartite_sets(triplets_text, n):
    from collections import defaultdict
    from itertools import combinations
    verb_to_obj = defaultdict(set)
    obj_to_verb = defaultdict(set)
    verbs = set()
    objects = set()
    for (_, verb, obj) in triplets_text.keys():
        if obj not in [17, 18, 19, 20, 21, 22] and verb not in [58]:
            verb_to_obj[verb].add(obj+1000)
            obj_to_verb[obj+1000].add(verb)
            verbs.add(verb)
            objects.add(obj+1000)
    valid_sets = []
    for verb_set in combinations(verbs, n):
        common_objects = set.intersection(*[verb_to_obj[v] for v in verb_set])
        if len(common_objects) >= n:
            for obj_set in combinations(common_objects, n):
                if all(len(obj_to_verb[obj].intersection(verb_set)) == n for obj in obj_set):
                    valid_sets.append((set(verb_set), set(obj_set)))
    return valid_sets

