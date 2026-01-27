# CrossSight👁️📝 - Image Captioning
A Multimodal Image Captioning System Built From Scratch using PyTorch

CrossSight is an End-to-End Image Captioning System that learns to **see** images and **generates** captions using CNN Encoder and Transformer Based Auto-Regressive Decoder.

The entire architecture is built from scratch using PyTorch , from attention mechanisms - to inference-time decoding , and is Deployed as an interactive web demo using Gradio. 

## 💻 Demo: 

HuggingFace Spaces: https://roshan454-crosssight.hf.space/?__theme=system&deep_link=6dMURe2Uxjw

This Demo Generates **token-by-token**  animated output captions which is similar to Large Language models generate text in real time.

(Demo_GIF)

## 🧐📝 Problem Statement: 

Image Captioning is a challenging multi-modal task that requires: 
- Extracting Visual Information from the image.
- Align the Visual Information with language.
- Generate Grammatically Correct and contextually relevant captions.
- Ensuring Autoregressive Generation.

Unlike classification tasks, captioning requires **sequence modeling**, **cross-attention**, and careful handling of training vs inference behavior.

## 🏛️🏗️System Architecture:


