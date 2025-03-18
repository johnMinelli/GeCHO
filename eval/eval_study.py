import os

import cv2
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from torchmetrics.image import FrechetInceptionDistance
from torchvision.datasets.folder import is_image_file
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from transformers import CLIPModel, CLIPProcessor

from data.hicodet.categories import roles, object_categories, triplets_text

DEVICE = "cuda"


# Function to check if a file is an image
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


# Custom Dataset class
class ImageFolderDataset(Dataset):
    def __init__(self, folder):
        self.images = [os.path.join(dirpath, filename) for dirpath, _, filenames in os.walk(folder) for filename in
                       filenames if is_image_file(filename)]
        self.transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = [self.transform(Image.open(self.images[i]).convert('RGB')) for i in idx]
        return {"im": torch.stack(images)}


def compute_fid(real_images_dir, synth_images_dir, batch_size=128, label=""):
    metric_fid = FrechetInceptionDistance(normalize=True).to(torch.device(DEVICE))

    # Create datasets and dataloaders
    real_dataset = ImageFolderDataset(real_images_dir)
    synth_dataset = ImageFolderDataset(synth_images_dir)

    real_dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    synth_dataloader = DataLoader(synth_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate real images
    progress_bar = tqdm(range(len(real_dataloader)), desc="Real Images")
    for real_images in real_dataloader:
        metric_fid.update(real_images["im"].to(DEVICE), real=True)
        progress_bar.update(1)

    # Evaluate synthetic images
    progress_bar = tqdm(range(len(synth_dataloader)), desc="Synthetic Images")
    for synth_images in tqdm(synth_dataloader, desc="Synthetic Images"):
        metric_fid.update(synth_images["im"].to(DEVICE), real=False)
        progress_bar.update(1)

    fid_value = metric_fid.compute()  # Inception, layer size 2048
    metric_fid.reset()

    print(f"{label}: Computed FID: {fid_value:.4f}")


def compute_clip_visual(real_images_dir, synth_images_dir, label="", limit=None):
    # Load the CLIP model and processor
    clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    scores = []
    distances = []

    # Get the list of synthetic images
    synth_images = [os.path.join(dirpath, filename).replace(synth_images_dir+'/', '')  for dirpath, _, filenames in os.walk(synth_images_dir) for filename in filenames if is_image_file(filename)]

    i = 0
    for synth_image_file in tqdm(synth_images, desc="Computing CLIP Visual Distances", total=None if not limit else limit+1):
        if limit and i>limit: break
        # Construct the corresponding real image filename
        real_image_file = synth_image_file.rsplit('_', 1)[0]+'.jpg'
        real_image_path = os.path.join(real_images_dir, real_image_file)

        # Load images
        synth_image = Image.open(os.path.join(synth_images_dir, synth_image_file)).convert('RGB')
        real_image = Image.open(real_image_path).convert('RGB') if os.path.exists(real_image_path) else None

        if real_image is None:
            print(f"Warning: Real image not found for {synth_image_file}")
            continue  # Skip if the real image doesn't exist

        # Prepare images for CLIP
        inputs = clip_processor(images=[synth_image, real_image], return_tensors="pt", padding=True).to(DEVICE)

        # Get CLIP embeddings
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)  # Get features for both images
        # Extract features
        synth_features = features[0]  # Features for the synthetic image
        real_features = features[1]  # Features for the real image
        scores.append((synth_features @ real_features.T).item())

        # Normalize the features
        synth_features /= synth_features.norm(dim=-1, keepdim=True)
        real_features /= real_features.norm(dim=-1, keepdim=True)
        distances.append(1 - (synth_features @ real_features.T).item())
        i+=1

    print(f"{label}: Average CLIP Visual distance: {np.mean(scores)} {np.mean(distances)}")


def compute_clip_textual(real_images_dir, prompt_file, synth_images_dir=None, apply_masking=False, filter_by_single_instance=True, label="", limit=None):
    import json
    import glob
    # Load the CLIP model and processor
    clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    scores = []
    distances = []

    def apply_mask(image, mask):
        image_np = np.array(image)
        mask_np = np.array(mask)
        mask_np = cv2.threshold(mask_np,50,1, cv2.THRESH_BINARY)[1]
        masked_image_np = image_np * mask_np[:, :, np.newaxis]  # Expand mask to match image channels

        return Image.fromarray(masked_image_np)


    with open(prompt_file, 'r') as f:
        lines = f.readlines()

    for i,l in enumerate(lines):
        data = json.loads(l.strip())
        data['conditioning'] = data['conditioning'].split('/')[-1].rsplit('_', 1)[0]+"_"+str(i)+"."+data['conditioning'].split('.')[-1]
        lines[i] = data

    if filter_by_single_instance:
        filtered_lines = []
        last_add = ""
        for l in lines:
            if l['conditioning'].split('/')[-1].rsplit('_', 1)[0] != last_add:
                filtered_lines.append(l)
                last_add = l['conditioning'].split('/')[-1].rsplit('_', 1)[0]
            elif last_add != "":
                filtered_lines.pop()
                last_add = ""
        lines = filtered_lines

    i=0
    for data in tqdm(lines, desc="Computing CLIP Textual Distances", total=None if not limit else limit+1):
        if limit and i>limit: break
        mask_image_path = os.path.join(real_images_dir, data['mask'])
        if synth_images_dir is not None:
            image_path = os.path.join(synth_images_dir, data['conditioning'])
        else:
            image_path = os.path.join(real_images_dir, data['target'])

        prompt = data['prompt']
        
        # Find all matching synthetic images
        # synth_image_pattern = os.path.join(synth_images_dir, data['conditioning'].split('/')[-1].rsplit('_', 1)[0] + "_*")
        # synth_images = glob.glob(synth_image_pattern)
        # if not synth_images:
        #    continue  # Skip if no synthetic images exist

        # Load the real image and mask it
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
        try:
            image = Image.open(image_path).convert('RGB')
            if apply_masking:
                image = apply_mask(image, mask_image)
            inputs = clip_processor(images=[image], text=[prompt], return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                image_features = clip_model.get_image_features(inputs.pixel_values)
                text_features = clip_model.get_text_features(inputs.input_ids)
            scores.append((image_features[0] @ text_features[0].T).item())
            # Normalize the features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            # Compute cosine distances
            distances.append(1 - (image_features[0] @ text_features[0].T).item())
        except Exception as e:
            pass
            # print(e)
        i+=1

    print(f"{label}: Average CLIP Textual distance: {np.mean(scores)}, {np.mean(distances)}")


def compute_clip_zs(real_images_dir, prompt_file, synth_images_dir=None, apply_masking=True, verb_class=False, objects_class=False, filter_by_single_instance=True, label="", limit=None):
    import json
    assert verb_class+objects_class==1, 'The ZS ACC can be computed either only for verbs or objects but no both.'
    # Load the CLIP model and processor
    clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    acc1 = 0
    acc5 = 0
    total_images = 0

    def apply_mask(image, mask):
        image_np = np.array(image)
        mask_np = np.array(mask)
        mask_np = cv2.threshold(mask_np,50,1, cv2.THRESH_BINARY)[1]
        masked_image_np = image_np * mask_np[:, :, np.newaxis]  # Expand mask to match image channels

        return Image.fromarray(masked_image_np)

    with open(prompt_file, 'r') as f:
        lines = f.readlines()
    
    for i,l in enumerate(lines):
        data = json.loads(l.strip())
        data['conditioning'] = data['conditioning'].split('/')[-1].rsplit('_', 1)[0]+"_"+str(i)+"."+data['conditioning'].split('.')[-1]
        lines[i] = data

    if filter_by_single_instance:
        filtered_lines = []
        last_add = ""
        for l in lines:
            if l['conditioning'].split('/')[-1].rsplit('_', 1)[0] != last_add:
                filtered_lines.append(l)
                last_add = l['conditioning'].split('/')[-1].rsplit('_', 1)[0]
            elif last_add != "":
                filtered_lines.pop()
                last_add = ""
        lines = filtered_lines

    i=0
    for data in tqdm(lines, desc=f"Computing CLIP Features for {'verb' if verb_class else 'object'} classification", total=None if not limit else limit+1):
        if limit and i>limit: break
        mask_image_path = os.path.join(real_images_dir, data['mask'])        
        if synth_images_dir is not None:
            image_path = os.path.join(synth_images_dir, data['conditioning'])
        else:
            image_path = os.path.join(real_images_dir, data['target'])

        gt_label = triplets_text[(1,data["role_category"],data["object_category"])]
        labels = [v for k,v in triplets_text.items() if k[2] == data['object_category']] if verb_class else [v for k,v in triplets_text.items() if k[1] == data['role_category']]

        # Load the real image and mask it
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
        try:
            image = Image.open(image_path).convert('RGB')
            if apply_masking:
                image = apply_mask(image, mask_image)

            inputs = clip_processor(images=[image], text=labels, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                image_features = clip_model.get_image_features(inputs.pixel_values)
                text_features = clip_model.get_text_features(inputs.input_ids)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ text_features.T  # Shape: (1, num_actions)

            # Get the top 5 predictions accuracy
            top_k_values, top_k_indices = torch.topk(similarities, 5)
            predicted_action = labels[top_k_indices[0, 0].item()]  # Top-1 prediction
            predicted_actions = [labels[idx.item()] for idx in top_k_indices[0]]  # Top-5 predictions

            # Update accuracy counters
            acc1 += (predicted_action == gt_label)
            acc5 += (gt_label in predicted_actions)
            total_images += 1

        except Exception as e:
            pass
            # print(f"Error processing image {image_path}: {e}")
        i+=1

    # Calculate final accuracy
    if total_images > 0:
        acc1 = (acc1 / total_images) * 100
        acc5 = (acc5 / total_images) * 100
    else:
        acc1 = acc5 = 0

    print(f"{label}: Top-1 Accuracy: {acc1:.2f}%")
    print(f"{label}: Top-5 Accuracy: {acc5:.2f}%")


if __name__ == "__main__":

    real_images_dir = './data/hicodet_formatted_test_edit'
    prompt_file = './data/hicodet_formatted_test_edit/prompt.json'
    edit_methods_to_eval = {
      "REAL": None, 
      "InstructPIX2PIX": './out/baseline/train_data_gen_hico_testset_baseline_instruct', 
      "AddSD": './out/baseline/train_data_gen_hico_testset_baseline_addsd', 
      "MagicBrush": './out/baseline/train_data_gen_hico_testset_baseline_magicbrush', 
      "MGIE": './out/baseline/train_data_gen_hico_testset_baseline_mgie',
      "MY": './out/baseline/train_data_gen_hico_testset_edit',
      "MY FULL": './out/baseline/train_data_gen_hico_testset_fullmask',
    }
    
    # compute_fid('./data/hicodet_formatted_test_edit/target', './data/hicodet_formatted_test_edit/removed_target', label='removed 2 real')
    # compute_clip_visual('./data/hicodet_formatted_test_edit/target', './data/hicodet_formatted_test_edit/removed_target', label='removed 2 real')
    # for name, path in edit_methods_to_eval.items():
    #     if path is not None: compute_fid(os.path.join(real_images_dir, 'target'), path, label=f"{name}")
    #     if path is not None: compute_clip_visual(os.path.join(real_images_dir, 'target'), path, label=f"{name}")
    #     compute_clip_textual(real_images_dir, prompt_file, synth_images_dir=path, label=f"{name}")
    #     compute_clip_zs(real_images_dir, prompt_file, synth_images_dir=path, filter_by_single_instance=True, objects_class=True, label=f"Object {name}")
    #     compute_clip_zs(real_images_dir, prompt_file, synth_images_dir=path, filter_by_single_instance=True, verb_class=True, label=f"Verb {name}")


    # Note: eval for inpainting methods not requires `filter_by_single_instance` to be True as the ambiguity concern refers only to editing methods (in inpainting all methods use an inpainting mask)  
    real_images_dir = './data/hicodet_formatted_test_inp'
    prompt_file = './data/hicodet_formatted_test_inp/prompt.json'
    inp_methods_to_eval = {
      "REAL": None,
      "MY": './out/train_data_gen_hico_testset_inp',
      "SD21-Inpainting": './out/train_data_gen_hico_testset_baseline_sdinp', 
      "PaintByExample ft25": './out/train_data_gen_hico_testset_baseline_pbe_ft25',
    }

    # for name, path in inp_methods_to_eval.items():
    #     if path is not None: compute_fid(os.path.join(real_images_dir, 'target'), path, label=f"{name}")
    #     if path is not None: compute_clip_visual(os.path.join(real_images_dir, 'target'), path, label=f"{name}")
    #     compute_clip_textual(real_images_dir, prompt_file, synth_images_dir=path, filter_by_single_instance=False, label=f"{name}")
    #     compute_clip_zs(real_images_dir, prompt_file, synth_images_dir=path, filter_by_single_instance=False, objects_class=True, label=f"Object {name}")
    #     compute_clip_zs(real_images_dir, prompt_file, synth_images_dir=path, filter_by_single_instance=False, verb_class=True, label=f"Verb {name}")
