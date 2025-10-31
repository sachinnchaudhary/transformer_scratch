class MultiHeadAttention(nn.Module):  
    
    def __init__(self, dim, head):  
            
        super().__init__()
        self.dim = dim 
        self.heads = head  
        self.head_dim  = dim // head 

        assert dim % head == 0

        self.q=  nn.Linear(dim,dim)
        self.k = nn.Linear(dim,dim)
        self.v=  nn.Linear(dim,dim)
        self.w_output = nn.Linear(dim,dim)

    def forward(self, q, k ,v, mask= None):  

        batch, seq_len, dim = q.shape
        batch, kv_len, dim = k.shape  

        q = self.q(q).reshape(batch, seq_len, self.heads, self.head_dim).transpose(1,2) 
        k = self.k(k).reshape(batch, kv_len, self.heads, self.head_dim).transpose(1,2)
        v = self.v(v).reshape(batch, kv_len, self.heads, self.head_dim).transpose(1,2)

        scores =  (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        
        if mask is not None:  
          scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim= -1)

        output = attn_weights @ v

        return output.transpose(2,1).contiguous().reshape(batch, seq_len, dim)
    
class FeedForward(nn.Module):
     
     def __init__(self, dim, dim_ff):  
          super().__init__()
          self.net = nn.Sequential(
              nn.Linear(dim, dim_ff),
              nn.GELU(),
              nn.Linear(dim_ff, dim)
          )
     def forward(self, x): 
         return self.net(x)

class EncoderLayer(nn.Module):   
      
      def __init__(self, dim, n_heads, ff_dim):   
          super().__init__()
          self.attn = MultiHeadAttention(dim, n_heads)
          self.ff = FeedForward(dim, ff_dim)
          self.ln1 = nn.LayerNorm(dim)
          self.ln2 = nn.LayerNorm(dim)
      
      def forward(self, x):   
           
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + self.ff(self.ln2(x))

        return x
    
class DecoderLayer(nn.Module):    
    
     def __init__(self,  dim, n_heads, ff_dim): 
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, n_heads)
        self.cross_attn = MultiHeadAttention(dim, n_heads)
        self.ff = FeedForward(dim, ff_dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
     
     def forward(self, x, enc_out, tgt_mask=None):
       x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), mask=tgt_mask)
       x = x + self.cross_attn(self.ln2(x), enc_out, enc_out)  
       x = x + self.ff(self.ln3(x))
       return x

class Transformer(nn.Module):    

     def __init__(self, vocab_size, dim, n_heads, n_layers, ff_dim, max_seq_len):
        super().__init__()  
        self.embedding = nn.Embedding(vocab_size, dim)
        self.register_buffer("pe",self._create_positional_encoding(max_seq_len, dim))

        self.encoder_layers = nn.ModuleList([EncoderLayer(dim, n_heads, ff_dim) for _ in range(n_layers)]) 
        self.decoder_layers = nn.ModuleList([DecoderLayer(dim, n_heads,ff_dim) for _ in range(n_layers)])
        self.output_proj = nn.Linear(dim, vocab_size)   

     def _create_positional_encoding(self, max_seq_len, dim):

         pe = torch.zeros(max_seq_len, dim)
         position = torch.arange(0, max_seq_len).unsqueeze(1)
         div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
         pe[:, 0::2] = torch.sin(position * div_term)
         pe[:, 1::2] = torch.cos(position * div_term)
         return pe.unsqueeze(0) 

     def forward(self, src, tgt):   

        src_emb = self.embedding(src) + self.pe[:, :src.shape[1], :]
        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)      
        
        tgt_emb = self.embedding(tgt) + self.pe[:, :tgt.shape[1], :]
        tgt_mask = self._create_causal_mask(tgt.shape[1]).to(tgt.device)
        
        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_mask)

        return self.output_proj(dec_out)   
     
     def _create_causal_mask(self, size): 
          mask = torch.tril(torch.ones(size, size))
          return mask.unsqueeze(0).unsqueeze(0)    
     

if __name__ == "__main__":
    vocab_size = 1000
    dim = 128
    n_heads = 8
    n_layers = 2
    ff_dim = 512
    max_seq_len = 100

    model = Transformer(vocab_size, dim, n_heads, n_layers, ff_dim, max_seq_len)   

    batch_size = 4
    src_len = 20
    tgt_len = 15

    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))   
    output = model(src, tgt)
    print(f"Output shape: {output.shape}")  # (4, 15, 1000)
