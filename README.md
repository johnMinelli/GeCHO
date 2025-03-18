# Contextual Inpainting

Implementation of generative inpainting methods adopted for contextual inpainting of images. The approach is subsequently  tested quantitatively by generating a synthetic dataset for Human-Object Interaction (HOI).

### Code Structure
```
<repo_source>
│
└───data
│   └───train: Train data (i.e. generated images, real images, inpainting backgrounds)
│   └───validation: Diffusion models validation files
│   └───{dataset}: Downloaded dataset 
│   └───{dataset}_formatted: Diffusion models training processed data  
│
└───docker: Setup the code repository in a docker machine
│
└───pipeline: Inference pipeline
│
└───preprocess: Script of preprocessing and detector/labeler utils
│
└───third: 3rd party libraries (e.g. Detectron, LLaVA)
│
└───utils: Utility functions, args parser and dataset class 
│
│   ctrl_train.py: Training script of the ControlNet inpainting 
│   lora_train.py: LoRA fine-tuning script for the UNet
│   lora_ctrl_train.py: LoRA fine-tuning script with the ControlNet
│   eval.py: Validation script for a diffusion model configuration 
│   generate_hoi.py: Script for large unsupervised data synthesis
│   requirements.txt  
│   README.md

```
---
# Scripts

## Preprocess
```
cd preprocess
python preprocess.py --action mask --input_path ./data/{dataset} --output_path ./data/{dataset}_formatted
python preprocess.py --action pose --input_path ./data/{dataset} --output_path ./data/{dataset}_formatted
python preprocess.py --action prompt --input_path ./data/{dataset} --output_path ./data/{dataset}_formatted --csv dataset_oih.csv
python preprocess.py --action clean --input_path ./data/{dataset}_formatted --csv dataset_oih.csv
python preprocess.py --action obj_mask --input_path ./data/{dataset}_formtted --csv dataset_oih.csv
```
- The preprocessing commands should be executed in sequence as presented. That said the data handling was somewhat messy during the project (due to back and forth and remote cleanup/training), so depending on the data you are using I suggest you to double check the scripts. The expected output should be:
```
<{dataset}_formatted>
│
└───mask: b/w masks of segmented people (people are 1 valued)
└───mask_obj: b/w masks of segmented objects (objects are 1 valued)
└───source: rgb background obtained from segmentation of the people in the image
└───target: original images
│   
│   prompt.json: File with instances of diffusion models training
```
- Processing the images we could encounter bad samples where the detection/labeling fails: `clean` script is the only one to filter the image (post right cropping and resizing to 512x512). 

## Launch ControlNet training
```
python ctrl_train.py --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-inpainting --output_dir=./out/ctrl_inp --train_data_dir ./data/{dataset}_formatted/ --validation_file=./data/validation/validation-{dataset}.json --learning_rate=1e-5 --num_train_epochs 250 --validation_steps 500 --checkpointing_steps 1000 --gradient_accumulation_steps 1 --train_batch_size 16
```

## Launch Unet fine-tuning
```
python lora_train.py --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-inpainting --output_dir=./out/unet_lora --train_data_dir ./data/{dataset}_formatted/ --validation_file ./data/validation/validation-{dataset}.json --learning_rate 0.000005 --rank 64 --alpha_rank 32 --lycorice_algo lora --num_train_epochs 75 --validation_steps 500 --checkpointing_steps 1000 --gradient_accumulation_steps 1 --train_batch_size 32
```

## Launch Unet fine-tuning on top of a trained ControlNet
```
python lora_ctrl_train.py --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-inpainting --controlnet_model_name_or_path={path}/checkpoint-{n}/controlnet --output_dir=./out/unet_lora-ctrl --train_data_dir ./data/{dataset}_formatted --validation_file ./data/validation/validation-{dataset}.json --learning_rate 0.000005 --rank 64 --alpha_rank 32 --lycorice_algo lora --num_train_epochs 50 --validation_steps 500 --checkpointing_steps 500 --gradient_accumulation_steps 1 --train_batch_size 16
```

## Notes
- Whenever possible keep a batch factor between 16 or 32 setting `train_batch_size` and `gradient_accumulation_steps`
- `lycorice_algo`, `rank` and `alpha_rank` are selected empirically
- The number of epochs is not indicative, as it depends upon the amount of data you have at disposal, check the validation images when the output seems right to you
- You can save some GB of VRAM adding `--enable_cpu_offload`
- Add `log` to enable external logging, specifying `wandb` or `tensorboard` as possible options
- For SD2.1 you can use `--pretrained_model_name_or_path=HieuPM/stable-diffusion-2-1-inpainting`

## Validate a trained model
Same validation as in training loop: useful to test different checkpoints.
```
python eval.py --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-inpainting --controlnet_model_name_or_path={path}/checkpoint-{n}/controlnet --lora_path={path}/checkpoint-{n} --evaluation_file ./data/validation/validation-{dataset}.json --num_validation_images 2
```

## Generate HOI data
Following our pipeline we synthesize new data by inpainting, starting from an HOI dataset (HICO-DET / VCOCO)
```
python generate_hoi.py --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-inpainting --controlnet_model_name_or_path={path}/checkpoint-{n}/controlnet --lora_path={path}/checkpoint-{n} --train_data_dir ./data/{dataset}_lora_formatted --batch_size 16
```
- Set `batch_size` and the flags `enable_cpu_offload`, `gradient_checkpointing` according to your hardware availability
- Masking method adopted can be changed modifying the parameters in the call of the inpainting pipeline
  - `dynamic_masking=False`
  - `aux_focus_prompt=batch["ctrl_txt"], dynamic_masking=True` 
  - `aux_focus_token=[HAND_TOKEN], dynamic_masking=True`
