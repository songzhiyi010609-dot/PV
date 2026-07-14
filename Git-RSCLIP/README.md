---
license: apache-2.0
tags:
- Vision
- Multi-model
- Vision-Language
- Remote-sensing
widget:
- src: >-
    https://huggingface.co/datasets/mishig/sample_images/resolve/main/cat-dog-music.png
  candidate_labels: playing music, playing sports
  example_title: Cat & Dog
---

# Git-RSCLIP-base

[[Git-RSCLIP]](https://arxiv.org/pdf/2501.00895) is pre-trained on the Git-10M dataset (a global-scale remote sensing image-text pair dataset, consisting of 10 million image-text pairs) at size 256x256, first released in [this repository](https://github.com/chen-yang-liu/Text2Earth). It employs a similar structure to [[google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224)]. 

This is a **base version**, the **large version** is here: [[**Git-RSCLIP-large**](https://huggingface.co/lcybuaa/Git-RSCLIP)]

## News 🔥
✅ 2025.06.01: **Git-RSCLIP** series downloads exceeded **60,000** times 🔥

## Intended uses & limitations

You can use the raw model for tasks like zero-shot image classification and image-text retrieval.


### How to use

#### Use Git-RSCLIP to get image features

```python
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch

model = AutoModel.from_pretrained("lcybuaa/Git-RSCLIP-base")
processor = AutoProcessor.from_pretrained("lcybuaa/Git-RSCLIP-base")

url = "https://github.com/Chen-Yang-Liu/PromptCC/blob/main/Example/B/train_000051.png?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
  image_features = model.get_image_features(**inputs)
```


#### zero-shot image classification:

```python
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch

model = AutoModel.from_pretrained("lcybuaa/Git-RSCLIP-base")
processor = AutoProcessor.from_pretrained("lcybuaa/Git-RSCLIP-base")

url = "https://github.com/Chen-Yang-Liu/PromptCC/blob/main/Example/B/train_000051.png?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

texts = ["a remote sensing image of river", "a remote sensing image of houses and roads"]
inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image) # these are the probabilities
top5_indices = torch.argsort(probs, descending=True)[:, :5].cpu().numpy()
top1_indices = top5_indices[:, 0]
print(f"the image 0 is '{top1_indices[0]}'")
```

For more code examples, we refer to the [documentation](https://huggingface.co/transformers/main/model_doc/siglip.html#).


## Training procedure

### Training data

Git-RSCLIP is pre-trained on the Git-10M dataset (a global-scale remote sensing image-text pair dataset, consisting of 10 million image-text pairs) [(Liu et al., 2024)](https://github.com/chen-yang-liu/Text2Earth).

### Preprocessing

Images are resized/rescaled to the same resolution (224x224) and normalized across the RGB channels with mean (0.5, 0.5, 0.5) and standard deviation (0.5, 0.5, 0.5).

Texts are tokenized and padded to the same length (64 tokens).


### BibTeX entry and citation info

```bibtex
@misc{liu2025text2earthunlockingtextdrivenremote,
      title={Text2Earth: Unlocking Text-driven Remote Sensing Image Generation with a Global-Scale Dataset and a Foundation Model},
      author={Chenyang Liu and Keyan Chen and Rui Zhao and Zhengxia Zou and Zhenwei Shi},
      year={2025},
      eprint={2501.00895},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.00895},
}
```