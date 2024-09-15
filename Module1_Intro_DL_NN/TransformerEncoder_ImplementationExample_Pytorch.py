from typing import Optional, Tuple, Union, Any  # Add this line
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


## 1. DEFINE FeedForward, LayerNorm, SkipConnection here
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            `d_model`: model dimension
            `d_ff`: hidden dimension of feed forward layer
            `dropout`: ropout rate, default 0.1
        """
        super(FeedForward, self).__init__() 
        self.d_model=d_model
        self.d_ff=d_ff
        self.dropout=dropout

        ## 1. DEFINE 2 LINEAR LAYERS AND DROPOUT HERE
        self.linear1=nn.Linear(d_model,d_ff)
        self.linear2=nn.Linear(d_ff,d_model)
        self.dropoutlayer=nn.Dropout(dropout)
       
       

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            `x`: shape (batch_size, max_len, d_model)
        Returns:
            same shape as input x
        """
        ## 2.  RETURN THE FORWARD PASS 
        x=self.linear1(x)
        x=F.relu(x)
        x=self.dropoutlayer(x)
        x=self.linear2(x)
        return x
      

class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        # features = d_model
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b

  

## 2. DEFINE ScaledDotProductAttention and MultiHeadAttention here

class ScaledDotProductAttention(nn.Module):
  def __init__(self):
    super(ScaledDotProductAttention, self).__init__()
  
  def forward(self,
              query: torch.FloatTensor,
              key: torch.FloatTensor,
              value: torch.FloatTensor,
              mask:Optional[torch.ByteTensor] = None,
              dropout: Optional[nn.Dropout] = None) -> Tuple[torch.Tensor,Any]:
      """
      Args:
          query: shape(batch_size, n_heads, max_len, d_q)
          key: shape(batch_size, n_heads,max_len, d_q)
          value: shape(batch_size, n_heads, max_len, d_q)
          mask: shape(batch_size, 1,1, max_len)
      Returns:
          weighted value: shape(batch_size, n_heads, max_len, d_v)
          weighted matrix: shape(batch_size, n_heads, max_len, max_len)
      """
      d_k=query.size(-1)
      scores=torch.matmul(query, key.transpose(-2,-1))/math.sqrt(d_k)
      if mask is not None:
        scores=scores.masked_fill(mask.eq(0), -1e9)
      p_attn=F.softmax(scores, dim=-1)
      if dropout is not None:
        p_attn=dropout(p_attn)
      return torch.matmul(p_attn,value), p_attn


class MultiHeadAttention(nn.Module):
  def __init__(self, n_heads: int,
                     d_model: int,
                     dropout: float = 0.1):
      super(MultiHeadAttention, self).__init__()
      assert d_model % n_heads == 0
      self.d_k = d_model // n_heads
      self.h = n_heads
      
      self.linears = nn.ModuleList(
        [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)]
        )
      self.spda = ScaledDotProductAttention()
      self.attn = None
      self.dropout = nn.Dropout(p = dropout)
  
  def forward(self, query: torch.FloatTensor,
                    key: torch.FloatTensor,
                    value: torch.FloatTensor,
                    mask: Optional[torch.ByteTensor] = None) -> torch.FloatTensor:
                      
      if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(1)
      
      batch_size = query.size(0)
      
      query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1,2)
                          for  l, x in zip(self.linears, (query, key, value))
      ]
      
      x, self.attn = self.spda(query, key, value, mask = mask)
      
      x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h*self.d_k)
      return self.linears[-1](x)


class SkipConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size: int, dropout: float):
        super(SkipConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                x: torch.FloatTensor, 
                sublayer: Union[MultiHeadAttention, FeedForward]
                ) -> torch.FloatTensor:
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))    
        
class EncoderLayer(nn.Module):
    """Encoder  layer"""

    def __init__(self, size: int, self_attn: MultiHeadAttention, feed_forward: FeedForward, dropout: float):
        super(EncoderLayer, self).__init__()
        ## 3. EncoderLayer subcomponents
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([copy.deepcopy(SkipConnection(size, dropout)) for _ in range(2)])
        self.size = size
       

    def forward(self, x: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        ## 4. EncoderLayer forward pass
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer: EncoderLayer, N: int):
        super(Encoder, self).__init__()
        ## 5. Encoder subcomponents
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.Norm=LayerNorm(layer.size)
        


    def forward(self, x: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        """Pass the input (and mask) through each layer in turn."""
        ## 6. Encoder forward pass
        for layer in self.layers:
          x = layer(x, mask)
        return self.norm(x)


class TransformerEncoder(nn.Module):
    """The encoder of transformer
    Args:
        `n_layers`: number of stacked encoder layers
        `d_model`: model dimension
        `d_ff`: hidden dimension of feed forward layer
        `n_heads`: number of heads of self-attention
        `dropout`: dropout rate, default 0.1
    """

    def __init__(self, d_model: int, d_ff: int, n_heads: int = 1, n_layers: int = 1,
                 dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        self.multi_headed_attention = MultiHeadAttention(n_heads, d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.encoder_layer = EncoderLayer(d_model, self.multi_headed_attention, self.feed_forward, dropout)
        self.encoder = Encoder(self.encoder_layer, n_layers)
        self.reset_parameters()
      

    def forward(self, x: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        return self.encoder(x, mask)