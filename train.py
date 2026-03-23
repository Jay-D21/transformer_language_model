import torch
from transformer import SimpleTransformer

if __name__ == "__main__":
    ntokens = 1000 # vocabulary size
    emsize = 200 # embedding dimension
    nhid = 200 # dimension of feedforward
    nlayers = 2 # number of layers
    nhead = 2 # number of attention heads
    
    model = SimpleTransformer(ntokens, emsize, nhead, nhid, nlayers)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters.")
    
    # Dummy input (seq_len, batch_size)
    dummy_input = torch.randint(0, ntokens, (35, 20))
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
