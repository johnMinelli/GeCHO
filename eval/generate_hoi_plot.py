import json
import os
from typing import Iterable

import cv2
import numpy as np
from lycoris import create_lycoris, LycorisNetwork
from peft import set_peft_model_state_dict, LoraConfig
from safetensors.torch import load_file
from torchvision.transforms.functional import pil_to_tensor

import wandb
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPTextModelWithProjection
from diffusers import ControlNetModel, PNDMScheduler, DDPMScheduler, UniPCMultistepScheduler, DDIMScheduler

from data.hicodet.categories import roles, object_categories
from eval.plot_categories import triplets_text_teaser, llava_query_teaser
from utils.dataset import ProcDataset, DiffuserDataset

from utils.data_utils import proc_collate_fn
from pipeline.pipeline_stable_diffusion_controlnet_inpaint import StableDiffusionControlNetImg2ImgInpaintPipeline
from utils.dataset import SynthDataset
from utils.parser import Eval_args

import torch
import csv
from PIL import Image, ImageDraw
from preprocess.llava_predictor import Predictor

llava_pred = Predictor()
llava_pred.setup()

def save_locally(log, out_path, out_json):
    # save locally 
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        with open(os.path.join(out_path, out_json), 'w') as file:
            file.write('[')

    image = log["image"]
    # draw = ImageDraw.Draw(image)
    # 
    # # Draw bounding boxes for subjects
    # for subject_bbox in log["hoi_subjects"]:
    #     draw.rectangle(subject_bbox, outline="red", width=2)
    # # Draw bounding boxes for objects
    # for object_bbox in log["hoi_objects"]:
    #     draw.rectangle(object_bbox, outline="blue", width=2)
    # 
    # for hoi_ann in log["hoi_annotation"]:
    #     draw.rectangle(log["annotations"][hoi_ann["subject_id"]]["bbox"], outline="green", width=2)
    #     draw.rectangle(log["annotations"][hoi_ann["object_id"]]["bbox"], outline="green", width=2)
    # save image
    filename = os.path.splitext(log["source_filename"])[0] + "_" + str(log["id"]) +f"_{roles[log['role_category']]}_{object_categories[log['object_category']]}" + os.path.splitext(log["source_filename"])[1]
    image.save(os.path.join(out_path, filename))

    json_line = {"file_name": filename, "new_triplet":allowed_triplets[(log["subject_category"], log["role_category"], log["object_category"])], "new_triplet_category":(log["subject_category"], log["role_category"], log["object_category"]), "new_prompt":log["prompt"]}
    with open(os.path.join(out_path, out_json), 'a') as outfile:
        json.dump(json_line, outfile)
        outfile.write(',\n')


args = Eval_args().parse_args()
# init ControlNet(s)
controlnet = None
controlnet_text_encoder = None
num_controlnets = 0
class_cond = False
if args.controlnet_model_name_or_path is not None:
    controlnet = [ControlNetModel.from_pretrained(net_path, torch_dtype=torch.float32) for net_path in args.controlnet_model_name_or_path]
    for net_path, net in zip(args.controlnet_model_name_or_path, controlnet):
        w_dict = load_file(os.path.join(net_path, "diffusion_pytorch_model.safetensors"))
        if w_dict.get("class_embedding.weight", None) is not None:
            class_cond = True
            net.class_embedding = torch.nn.Embedding(4, 1280)
        net.load_state_dict(w_dict)
    num_controlnets = len(controlnet)
    if num_controlnets == 1:
        controlnet = controlnet[0]
    controlnet_text_encoder = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(torch.device("cuda"))

# init pipeline
pipeline = StableDiffusionControlNetImg2ImgInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        controlnet_text_encoder=controlnet_text_encoder,
        controlnet_prompt_seq_projection=False,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=torch.float32
    ).to(torch.device("cuda"))
pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

# init LoRA modules
if args.lora_path is not None:
    LycorisNetwork.apply_preset({"target_module": ["ResnetBlock2D"]})  # by module (e.g. Attention, Transformer2DModel, ResnetBlock2D) or by wildcard (e.g. {"target_name": [".*attn.*"]})
    lyco = create_lycoris(pipeline.unet, 1.0, linear_dim=64, linear_alpha=32, algo="lora").cuda()
    lyco.apply_to()
    lyco_state = torch.load(f"{args.lora_path}/lycorice.ckpt", map_location=torch.device("cuda"))
    lyco.load_state_dict(lyco_state, strict=False)
    lyco.cuda()
    lyco.merge_to(1.0)

# memory optimization
if args.enable_cpu_offload:
    pipeline.enable_model_cpu_offload()
if args.enable_xformers_memory_efficient_attention:
    pipeline.enable_xformers_memory_efficient_attention()

# randomization
if args.seed is None:
    generator = None
else:
    generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)
    np.random.seed(args.seed)

# data
fix = "teaser"
allowed_triplets = triplets_text_teaser
llava_query_template = llava_query_teaser
dataset = SynthDataset(args.train_data_dir[0], 'prompt.json', 10000, None,512, dilated_conditioning_mask=False)  
dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=0, drop_last=True, collate_fn=proc_collate_fn)

for gen_id, batch in enumerate(dataloader):
    gen_start_index = 0
    old_obj_category_k, old_obj_category_v = int(batch["object_category"][0]), object_categories[int(batch["object_category"][0])]
    old_role_category_k, old_role_category_v = int(batch["role_category"][0]), roles[int(batch["role_category"][0])]
    
    
    for new_obj_category_k, new_obj_category_v in object_categories.items():
        for new_role_category_k, new_role_category_v in roles.items():
            if new_obj_category_k!=1 and (1, new_role_category_k, new_obj_category_k) in allowed_triplets:
                query = llava_query_template.format(
                    triplet_text=allowed_triplets[(1, new_role_category_k, new_obj_category_k)].replace("a photo of ", ""),
                    object=new_obj_category_v,
                    verb=new_role_category_v
                )
    
                prompt = llava_pred.predict(os.path.join(args.train_data_dir[0], batch["image_name"][0].replace('mask', 'source')), query, 0.7, 0.2, 512)
                prompt = (prompt[:-1] if prompt.endswith('.') else prompt).replace("\"", "").replace("'", "")
                batch["txt"][0] = prompt
                batch["object_category"][0] = new_obj_category_k
                batch["role_category"][0] = new_role_category_k
    
                with torch.no_grad():
                    with torch.autocast(f"cuda", dtype=torch.float32):
                        pred_images = pipeline(prompt=batch["txt"], controlnet_prompt=batch["ctrl_txt"], negative_prompt=[batch["neg_txt"][0]+(f", {old_obj_category_v}" if old_obj_category_v!=new_obj_category_v else "")], aux_focus_token=batch["ctrl_txt"], dynamic_masking=True, # recompute_first_step=True, 
                                              image=batch["image"], mask_image=batch["mask"], conditioning_image=batch["conditioning"], height=512, width=512, 
                                              strength=1.0, controlnet_conditioning_scale=0.9, num_inference_steps=45, guidance_scale=7.5, self_guidance_scale=0,
                                              num_images_per_prompt=4, guess_mode=True, generator=generator, gradient_checkpointing=args.gradient_checkpointing, return_dict=False)            
                # log&save final result
                for j, pred_image in enumerate(pred_images):
                    i=0
                    log = {"id":gen_id+(gen_start_index*4)+j, "source": batch["image"][i], "source_filename":os.path.basename(batch["image_name"][i]), 
                           "image": pred_image, "prompt": batch["txt"][i], "control": batch["ctrl_txt"][i][-1] if type(controlnet)==list else batch["ctrl_txt"][i],
                           "subject_category": int(batch["subject_category"][i]), "object_category": int(batch["object_category"][i]), "role_category": int(batch["role_category"][i]),
                           "old_object_category": old_obj_category_k, "old_role_category": old_role_category_k,
                           "hoi_annotation":json.loads(batch["hoi_annotation"][i]), "annotations": json.loads(batch["annotations"][i])}
            
                    category = (int(batch["subject_category"][i]), int(batch["role_category"][i]), int(batch["object_category"][i]))
                    save_locally(log, out_path=os.path.join("./out", f"train_data_gen_{fix}"), out_json="gen.json")

                gen_start_index += 1
