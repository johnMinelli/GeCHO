import json
import os
import sys
from contextlib import nullcontext

import numpy as np
import torchvision
from pytorch_lightning import seed_everything
import torchvision.transforms as T
from torchvision.transforms.functional import pil_to_tensor

import wandb
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPTextModelWithProjection, CLIPVisionModel, AutoTokenizer, CLIPImageProcessor
from diffusers import ControlNetModel, PNDMScheduler, DDPMScheduler, UniPCMultistepScheduler, DDIMScheduler
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

from data.hicodet.categories import triplets_text, object_categories

from utils.data_utils import proc_collate_fn
from utils.dataset import SynthDataset
from utils.parser import Eval_args
from utils.utils import PROJECT_HOME

import torch
from PIL import Image, ImageDraw


DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
DEFAULT_IM_START_TOKEN = '<im_start>'
DEFAULT_IM_END_TOKEN = '<im_end>'


class MGIE_Pipeline():
    def __init__(self):
        from third.mgie.mgie_llava import MGIELlavaLlamaForCausalLM
        from llava.conversation import conv_templates
        from llava.model import *

        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained('timbrooks/instruct-pix2pix', torch_dtype=torch.float16, safety_checker=None).to('cuda')
        self.pipe.unet.load_state_dict(torch.load(f'{PROJECT_HOME}/third/mgie/_ckpt/mgie_7b/unet.pt', map_location='cpu'))
        self.tokenizer = AutoTokenizer.from_pretrained(f'{PROJECT_HOME}/third/mgie/_ckpt/LLaVA-7B-v1')
        self.tokenizer.padding_side = 'left'
        self.tokenizer.add_tokens(['[IMG0]', '[IMG1]', '[IMG2]', '[IMG3]', '[IMG4]', '[IMG5]', '[IMG6]', '[IMG7]'], special_tokens=True)
        self.model = MGIELlavaLlamaForCausalLM.from_pretrained(f'{PROJECT_HOME}/third/mgie/_ckpt/LLaVA-7B-v1', low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()
        self.model.resize_token_embeddings(len(self.tokenizer))
        ckpt = torch.load(f'{PROJECT_HOME}/third/mgie/_ckpt/mgie_7b/mllm.pt', map_location='cpu')
        self.model.load_state_dict(ckpt, strict=False)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.model.config.mm_vision_tower, torch_dtype=torch.float16)
        self.device = self.pipe.device
        mm_use_im_start_end = getattr(self.model.config, 'mm_use_im_start_end', False)
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    
        vision_tower = CLIPVisionModel.from_pretrained(self.model.get_model().vision_tower[0].config.name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
        self.model.get_model().vision_tower[0] = vision_tower
        vision_config = vision_tower.config
        vision_config.im_patch_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        self.image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
        self.conv_template = conv_templates["vicuna_v1"]
    
        self.model.eval()
        self.EMB = ckpt['emb'].cuda()
        with torch.inference_mode():
            self.NULL = self.model.edit_head(torch.zeros(1, 8, 4096).half().to('cuda'), self.EMB)

    def remove_alter(s):  # hack expressive instruction
        if 'ASSISTANT:' in s: s = s[s.index('ASSISTANT:') + 10:].strip()
        if '</s>' in s: s = s[:s.index('</s>')].strip()
        if 'alternative' in s.lower(): s = s[:s.lower().index('alternative')]
        if '[IMG0]' in s: s = s[:s.index('[IMG0]')]
        s = '.'.join([s.strip() for s in s.split('.')[:2]])
        if s[-1] != '.': s += '.'
        return s.strip()

    def __call__(self, prompts, image, generator=None):
        img = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()
        
        conv_prompts = []
        for prompt in prompts:
            prompt = f"what will this image be like if '{prompt}'" + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len + DEFAULT_IM_END_TOKEN
            conv = self.conv_template.copy()
            conv.append_message(conv.roles[0], prompt), conv.append_message(conv.roles[1], None)
            conv_prompts.append(conv.get_prompt())
        conv_prompts = self.tokenizer(conv_prompts, padding=True, return_tensors='pt')
        txt, mask = conv_prompts['input_ids'].cuda(), conv_prompts['attention_mask'].cuda()

        with torch.inference_mode():
            out = self.model.generate(txt, images=img.half(),
                                 attention_mask=mask, do_sample=False, max_new_tokens=96,
                                 num_beams=1, no_repeat_ngram_size=3, return_dict_in_generate=True,
                                 output_hidden_states=True)
            out, hid = out['sequences'], torch.cat([x[-1] for x in out['hidden_states']], dim=1)

            positions = [min(o.index(32003) - 1 if 32003 in o else len(h) - 9, len(h) - 9) for o, h in zip(out, hid)]
            indices = torch.stack([torch.arange(p, p + 8).long() for p in positions]).cuda()
            hid = hid.gather(1, indices.unsqueeze(2).expand(*indices.shape, hid.size(2)))

            # out = remove_alter(self.tokenizer.decode(out))
            emb = self.model.edit_head(hid, self.EMB)
            res = self.pipe(image=image, prompt_embeds=emb, negative_prompt_embeds=self.NULL.expand(emb.size(0),*self.NULL.shape[1:]), generator=generator)

            return res


class PaintByExamplePipeline:

    def __init__(self, config_path, model_ckpt, device=None):
        
        """Initialize the pipeline with model and configuration.

        Args:
            config_path (str): Path to the model config file
            model_ckpt (str): Path to the model checkpoint
            device (str, optional): Device to run the model on. Defaults to CUDA if available.
        """
        sys.path.append(f"{PROJECT_HOME}/third/pbe")

        from ldm.models.diffusion.ddim import DDIMSampler
        from ldm.models.diffusion.plms import PLMSSampler
        from omegaconf import OmegaConf

        self.device = device

        # Load model
        config = OmegaConf.load(config_path)
        self.model = self.load_model_from_config(config, model_ckpt)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Initialize samplers
        self.sampler = PLMSSampler(self.model)

        # Setup transforms
        self.image_transform = self.get_tensor()
        self.clip_transform = self.get_tensor_clip()

    @staticmethod
    def load_model_from_config(config, ckpt):
        """Load model from config and checkpoint."""
        from ldm.util import instantiate_from_config

        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        model.load_state_dict(sd, strict=False)
        return model

    @staticmethod
    def get_tensor(normalize=True, toTensor=True):
        """Get transform for regular images."""
        transform_list = []
        if toTensor:
            transform_list += [torchvision.transforms.ToTensor()]
        if normalize:
            transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return torchvision.transforms.Compose(transform_list)

    @staticmethod
    def get_tensor_clip(normalize=True, toTensor=True):
        """Get transform for CLIP images."""
        transform_list = []
        if toTensor:
            transform_list += [torchvision.transforms.ToTensor()]
        if normalize:
            transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                (0.26862954, 0.26130258, 0.27577711))]
        return torchvision.transforms.Compose(transform_list)

    def extract_reference_from_mask(self, image, mask, padding=10):
        """Extract reference image from the target image using the mask's active area.

        Args:
            image: numpy array of shape (H, W, C)
            mask: numpy array of shape (H, W)
            padding: padding around the bounding box

        Returns:
            PIL.Image: Cropped reference image
        """
        # Find the bounding box of the white region in the mask
        white_pixels = np.where(mask > 127)
        if len(white_pixels[0]) == 0:  # Handle empty mask
            return Image.fromarray(image)

        # Get bounding box coordinates
        y_min, y_max = white_pixels[0].min(), white_pixels[0].max()
        x_min, x_max = white_pixels[1].min(), white_pixels[1].max()

        # Extract the reference region
        reference = image[y_min:y_max, x_min:x_max]
        return Image.fromarray(reference)

    def __call__(self, image: Image.Image, mask: Image.Image, reference_mask: Image.Image, ddim_steps: int = 50, scale: float = 5.0, h: int = 512, w: int = 512):
        """Process the images and return the result.

        Args:
            image (PIL.Image): Input image to be processed
            mask (PIL.Image): Mask image
            ddim_steps (int, optional): Number of DDIM sampling steps. Defaults to 50.
            scale (float, optional): Unconditional guidance scale. Defaults to 1.0.
            h (int, optional): Output height. Defaults to 512.
            w (int, optional): Output width. Defaults to 512.

        Returns:
            PIL.Image: Processed image
        """
        # Prepare images
        image_tensor = self.image_transform(image).unsqueeze(0)
        reference = self.extract_reference_from_mask(np.array(image), np.array(reference_mask.convert("L")))
        ref_tensor = self.clip_transform(reference.convert("RGB").resize((224, 224))).unsqueeze(0)

        # Process mask
        mask = np.array(mask.convert("L"))[None, None]
        mask = 1 - mask.astype(np.float32) / 255.0
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask_tensor = torch.from_numpy(mask)

        # Prepare inputs
        inpaint_image = image_tensor * mask_tensor
        test_model_kwargs = {'inpaint_mask': mask_tensor.to(self.device), 'inpaint_image': inpaint_image.to(self.device)}
        ref_tensor = ref_tensor.to(self.device)

        with torch.no_grad():
            with torch.autocast("cuda") if self.device == "cuda" else nullcontext():
                with self.model.ema_scope():
                    # Get conditioning
                    uc = None if scale == 1.0 else self.model.learnable_vector
                    c = self.model.get_learned_conditioning(ref_tensor.to(torch.float16))
                    c = self.model.proj_out(c)

                    # Process inpainting inputs
                    z_inpaint = self.model.encode_first_stage(test_model_kwargs['inpaint_image'])
                    z_inpaint = self.model.get_first_stage_encoding(z_inpaint).detach()
                    test_model_kwargs['inpaint_image'] = z_inpaint
                    test_model_kwargs['inpaint_mask'] = T.Resize([z_inpaint.shape[-2], z_inpaint.shape[-1]])(test_model_kwargs['inpaint_mask'])

                    # Sample
                    shape = [4, h // 8, w // 8]  # assume f=8
                    samples, _ = self.sampler.sample(S=ddim_steps, conditioning=c, batch_size=1, shape=shape,
                        verbose=False, unconditional_guidance_scale=scale, unconditional_conditioning=uc, eta=0.0,
                        test_model_kwargs=test_model_kwargs)

                    # Decode
                    x_samples = self.model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()

                    # Convert to PIL
                    x_sample = 255. * x_samples[0]
                    return Image.fromarray(x_sample.astype(np.uint8))


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
    filename = os.path.splitext(log["source_filename"])[0] + "_" + str(log["id"]) + os.path.splitext(log["source_filename"])[1]
    image.save(os.path.join(out_path, filename))


args = Eval_args().parse_args()

pipe = pbe_pipe = PaintByExamplePipeline(config_path=f"{PROJECT_HOME}/third/pbe/configs/v1.yaml", model_ckpt=f"{PROJECT_HOME}/third/pbe/checkpoints/epoch=000023.ckpt", device="cuda")
# pipe = mgie_pipe = MGIE_Pipeline()
# pipe = magicbrush_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("vinesmsuic/magicbrush-paper", torch_dtype=torch.float16).to("cuda")  # "vinesmsuic/magicbrush-jul7"
# magicbrush_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(magicbrush_pipe.scheduler.config)
# pipe = instruct_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None).to("cuda")
# instruct_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(instruct_pipe.scheduler.config)

# randomization
if args.seed is None:
    generator = None
else:
    generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
    np.random.seed(args.seed)
    seed_everything(args.seed)
pipe.safety_checker = None

# data
fix = "hico_testset_baseline_pbe"
gen_start_index = 0
dataset = SynthDataset(args.train_data_dir[0], args.train_data_file, 20000, None, 512, dilated_conditioning_mask=False)  # num_images if args.num_validation_images is None else min(num_images, args.num_validation_images), cat_probabilities, 
dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=0, drop_last=True, collate_fn=proc_collate_fn)

for gen_id, batch in enumerate(dataloader):
    with torch.no_grad():
        prompts = [triplets_text[(int(s), int(r), int(o))].replace("a photo of a", f"add a {object_categories[int(o)]} and make the") for s,r,o in zip(batch["subject_category"], batch["role_category"], batch["object_category"])]
        pred_images = [pbe_pipe(image=batch["image"][0], mask=batch["mask"][0], reference_mask=batch["conditioning"][0])]
        # pred_images = mgie_pipe(prompts, image=batch["removed_image"], generator=generator).images
        # pred_images = magicbrush_pipe(prompts, image=batch["removed_image"], num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7, generator=generator).images
        # pred_images = instruct_pipe(prompts, image=batch["removed_image"], generator=generator).images
    torch.cuda.empty_cache()
    # log&save final result
    for i, pred_image in enumerate(pred_images):
        log = {"id":gen_start_index+(gen_id*args.batch_size)+i, "source_filename":os.path.basename(batch["image_name"][i]), "image": pred_image,
               "subject_category": int(batch["subject_category"][i]), "object_category": int(batch["object_category"][i]), "role_category": int(batch["role_category"][i])}

        save_locally(log, out_path=os.path.join("./out", f"train_data_gen_{fix}"), out_json="gen.json")
