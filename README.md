# CrossSight👁️📝 - Image Captioning
A Multimodal Image Captioning System Built From Scratch using PyTorch

CrossSight is an End-to-End Image Captioning System that learns to **see** images and **generates** captions using CNN Encoder and Transformer Based Auto-Regressive Decoder.

The entire architecture is built from scratch using PyTorch , from attention mechanisms - to inference-time decoding , and is Deployed as an interactive web demo using Gradio. 

## Demo💻: 

HuggingFace Spaces: https://roshan454-crosssight.hf.space/?__theme=system&deep_link=6dMURe2Uxjw

This Demo Generates **token-by-token**  animated output captions which is similar to Large Language models generate text in real time.

(Demo_GIF)

## Problem Statement🧐📝: 

Image Captioning is a challenging multi-modal task that requires: 
- Extracting Visual Information from the image.
- Align the Visual Information with language.
- Generate Grammatically Correct and contextually relevant captions.
- Ensuring Autoregressive Generation.

Unlike classification tasks, captioning requires **sequence modeling**, **cross-attention**, and careful handling of training vs inference behavior.

## System Architecture🏛️🏗️:


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

## Model Details⚙️🔎: 

#### Encoder: 
- Backbone: EfficientNet-B0(Pre-trained).
- Linear Projection Layer gives output as shape (B,1,512)

#### Decoder: 
- Transformer Decoder(6 layers , 8 heads).
- Uses Causal Masking to prevent seeing the future tokens.
- Cross-Attention with the Image Memory from the Encoder.

## Training 🛠️🔄: 

![Training vs Inference](Images_and_Diagrams/Training_vs._Inference_Final.drawio.png)

- Teacher Forcing is used during Training.
- Captions are shifted:
  - Input: "<START> a little girl riding a bike
  - Output: " a little girl riding a bike <STOP>" 
- CrossEntropy Loss is computed across all timestep.
- '<PAD>' tokens are ignored during loss computation.

## Inference 📝🤔💬:

![Autoregressive Generation](Images_and_Diagrams/Autoregressive_Caption_Generation_Final.drawio.png)

During Inference:
- Generation is started with <START> token.
- Model Predict One Token at a time.
- Each Token is appended to the input sequence.
- Generation stops at <END> or max_length reaches.

This replicates how GPT-style Language models generate text.

## Results📊:

#### Training Loss Curve📉:

![Training_Loss_Curve](Images_and_Diagrams/loss_curve_cross_sight_30_epochs.png)

**Observation🔬📝:**  
- Loss is steadily Decreasing overall.
- Some noise is expected due to small dataset size and Autoregressive sequence learning. 

### Qualitative Example(Seen Data):

![Prediction Example](Images_and_Diagrams/Prediction_Example.png)

**Prediction:**  
> a black dog and a brown dog are standing on the street

**Ground Truth Caption:**  
> a black dog and a white dog with brown spots are staring at each other in the street

**Observation🔬📝:** 
Even when exact wording differs:
- Model correctly identifies the main entities (two dogs)
- the scene (street), and their interaction.  
- shows attributes such as coat color and exact action.

### Qualitative Results(Unseen Data):

![Prediction Example Unseen1](Images_and_Diagrams/unseen_data.png)
**Example 1**

![Prediction Example Unseen2](Images_and_Diagrams/predicton_2.png)
**Example 2**

![Prediction Example Unseen3](Images_and_Diagrams/prediction_3.png)
**Example 3**

![Prediction Example Unseen3](Images_and_Diagrams/prediction_4.png)
**Example 4**














  





