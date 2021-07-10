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





# With the LM head in GPT2LM
from vit_gpt2.modeling_flax_vit_gpt2_lm import FlaxViTGPT2LMForConditionalGeneration
flax_vit_gpt2_lm = FlaxViTGPT2LMForConditionalGeneration.from_pretrained(
    '.',
)

logits = flax_vit_gpt2_lm(**inputs)[0]
preds = np.argmax(logits, axis=-1)
print('=' * 60)
print('Flax: Vit + modified GPT2LM')
print(preds)

# flax_vit_gpt2_lm.save_pretrained('.')

del flax_vit_gpt2_lm
