import sys, os

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

# Vit - as encoder
from transformers import ViTFeatureExtractor
from PIL import Image
import requests
import numpy as np

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
encoder_inputs = feature_extractor(images=image, return_tensors="jax")
pixel_values = encoder_inputs.pixel_values

# GPT2 / GPT2LM - as decoder
from transformers import ViTFeatureExtractor, GPT2Tokenizer

name = 'asi/gpt-fr-cased-small'
tokenizer = GPT2Tokenizer.from_pretrained(name)
decoder_inputs = tokenizer("mon chien est mignon", return_tensors="jax")

inputs = dict(decoder_inputs)
inputs['pixel_values'] = pixel_values
print(inputs)

# With new added LM head
from vit_gpt2.modeling_flax_vit_gpt2 import FlaxViTGPT2ForConditionalGeneration
flax_vit_gpt2 = FlaxViTGPT2ForConditionalGeneration.from_vit_gpt2_pretrained(
    'google/vit-base-patch16-224-in21k', 'asi/gpt-fr-cased-small'
)
logits = flax_vit_gpt2(**inputs)[0]
preds = np.argmax(logits, axis=-1)
print('=' * 60)
print('Flax: Vit + modified GPT2 + LM')
print(preds)

del flax_vit_gpt2

# With the LM head in GPT2LM
from vit_gpt2.modeling_flax_vit_gpt2_lm import FlaxViTGPT2LMForConditionalGeneration
flax_vit_gpt2_lm = FlaxViTGPT2LMForConditionalGeneration.from_vit_gpt2_pretrained(
    'google/vit-base-patch16-224-in21k', 'asi/gpt-fr-cased-small'
)

logits = flax_vit_gpt2_lm(**inputs)[0]
preds = np.argmax(logits, axis=-1)
print('=' * 60)
print('Flax: Vit + modified GPT2LM')
print(preds)

del flax_vit_gpt2_lm

# With PyTorch [Vit + unmodified GPT2LMHeadModel]
import torch
from transformers import ViTModel, GPT2Config, GPT2LMHeadModel

vit_model_pt = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
encoder_inputs = feature_extractor(images=image, return_tensors="pt")
vit_outputs = vit_model_pt(**encoder_inputs)
vit_last_hidden_states = vit_outputs.last_hidden_state

del vit_model_pt

inputs_pt = tokenizer("mon chien est mignon", return_tensors="pt")
inputs_pt = dict(inputs_pt)
inputs_pt['encoder_hidden_states'] = vit_last_hidden_states

config = GPT2Config.from_pretrained('asi/gpt-fr-cased-small')
config.add_cross_attention = True
gpt2_model_pt = GPT2LMHeadModel.from_pretrained('asi/gpt-fr-cased-small', config=config)

gp2lm_outputs = gpt2_model_pt(**inputs_pt)
logits_pt = gp2lm_outputs.logits
preds_pt = torch.argmax(logits_pt, dim=-1).cpu().detach().numpy()
print('=' * 60)
print('Pytorch: Vit + unmodified GPT2LM')
print(preds_pt)

del gpt2_model_pt
