import torch 
from torch import nn 
import timm 

### Image Encoder 

class ImageEncoder(nn.Module):
      def __init__(self,model_dim):
          super().__init__()

          self.cnn = timm.create_model(
              "efficientnet_b0",
              pretrained=True,
              num_classes=0
          )
          
          dummy = torch.randn(1,3,224,224)
          with torch.no_grad():
               out = self.cnn(dummy)
          in_features = out.shape[-1]
          self.projection = nn.Linear(in_features,model_dim)

      def forward(self,images):
          features = self.cnn(images)   # (B,1280)
          features = self.projection(features) #(B,512) 
          return features.unsqueeze(1)

### Caption Decoder 

class CaptionDecoder(nn.Module):
    def __init__(self,vocab_size,model_dim,context_length,num_heads,num_layers):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size,model_dim)
        self.pos_embedding = nn.Embedding(context_length,model_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model = model_dim ,
            nhead = num_heads,
            dim_feedforward = 4* model_dim,
            dropout = 0.2 ,
            batch_first=True 
        )
    
        self.decoder = nn.TransformerDecoder(decoder_layer,num_layers)
        self.vocab_proj = nn.Linear(model_dim,vocab_size)
    
    def forward(self,tokens, image_memory):
        B, T = tokens.shape 
        device = tokens.device 

        tok_emb = self.token_embedding(tokens)
        pos = torch.arange(T,device=device)
        pos_emb = self.pos_embedding(pos)

        x = tok_emb + pos_emb 

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)

        out = self.decoder(
            tgt=x,
            memory=image_memory,
            tgt_mask=causal_make
        )

        return self.vocab_proj(out) 

## Encoder + Decoder 

class ImageCaptioningModel(nn.Module):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder = encoder 
        self.decoder = decoder 
    
    def forward(self,image,tokens):
        image_memory = self.encoder(images)
        return self.decoder(tokens,image_memory)
