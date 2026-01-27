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


![System Architecture](Images_and_Diagrams/CrossSight-System-Architecture_Final.png)

This System follows a CNN Encoder + Transformer Decoder Architecture: 
#### Image Encoder:
- Image is passed to EfficientNetB0 to extract visual features.
- This Features are passed through a linear projection layer which projects to the Transformer's model dimension.
- The Output from the linear layer is treated as Image Memory which is used for **cross-attention** to generate next token.

#### Caption Decoder: 
- The input tokens are passed with token embeddings + positional embeddings.
- Transformer Decoder has Masked Self-Attention and Cross-Attention over the image memory.
- Linear Projection layer to vocabulary logits.

##⚙️🔎Model Details: 

#### Encoder: 
- Backbone: EfficientNet-B0(Pre-trained).
- Linear Projection Layer gives output as shape (B,1,512)
- Acts as a fixed visual-context for the decoder. (Like a writer describing a painting in the wall, here the painting is a fixed thing and his writing is autoregressive).

#### Decoder: 
- Transformer Decoder(6 layers , 8 heads).
- Uses Causal Masking to prevent seeing the future tokens.
- Cross-Attention with the Image Memory from the Encoder.

  





