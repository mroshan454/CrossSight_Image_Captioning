import torch 
import gradio as gr 
import time 
from model import ImageEncoder, CaptionDecoder , ImageCaptioningModel 
from vocab import vocab , idx_to_word 
import torchvision.transforms as T 

device = "cuda" if torch.cuda.is_available() else "cpu"


### Load Model 

model_dim = 512 
context_length = 20 

encoder = ImageEncoder(model_dim)
decoder = CaptionDecoder(
    vocab_size=len(vocab),
    model_dim=model_dim,
    context_length=context_length,
    num_heads=8,
    num_layers=6
)

model = ImageCaptioningModel(encoder,decoder)
model.load_state_dict(torch.load("Cross_Sight_30_Epochs.pt",map_location=device))
model.to(device)
model.eval()

# Image Preprocessing 

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], 
                std = [0.5,0.5,0.5])
])

## Animated Caption Generator 

def generate_caption(image,max_len=20):

    image = transform(image).unsqueeze(0).to(device)
    tokens = torch.tensor([[vocab["<START>"]]],device=device)

    caption_words = []

    with torch.inference_mode():
      for _ in range(max_len):
          logits = model(image,tokens)      # (1, T,V)
          next_id = logits[:,-1,:].argmax(dim=-1) # (1,)

          if next_id.item() == vocab["<END>"]:
             break 
          
          word = idx_to_word[next_id.item()]
          caption_words.append(word)

          # Animated Typing 
          yield " ".join(caption_words)

          tokens = torch.cat([tokens,next_id.unsqueeze(1)], dim=1)
          time.sleep(0.25) #Typing Speed 

## Gradio UI 

demo = gr.Interface(
    fn = generate_caption,
    inputs = gr.Image(type="pil",label = "Upload Image"),
    outputs = gr.Textbox(label="Generated Caption"),
    title = "Cross Sight👁️",
    description = "A Multimodal Image Captioning System Built from Scratch"
)

demo.launch() 
