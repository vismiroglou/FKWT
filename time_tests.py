import torch
from functools import partial
from argparse import ArgumentParser
import time
from tqdm import tqdm
from src.KWT import PreNorm, PostNorm, Attention, FeedForward
from torch import nn

class Fourier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fourier = partial(torch.fft.fftn, dim=(1,2))

    def forward(self, x):
        return self.fourier(x).real


def main(batch, fourier_seq, attention_seq):
    #end = 0
    # batch = torch.rand([int(args.batch_size), int(args.seq_len), int(args.embed_len)])
    # device = torch.device('cuda')
    #fourier_seq = fourier_seq.to(device)
    #attention_seq = attention_seq.to(device)
    #batch = batch.to(device)
    fourier_times = []
    attention_times = []
    for i in tqdm(range(1001)):
        start = time.time()
        outputs = fourier_seq(batch)
        end = time.time()-start
        fourier_times.append(end)
        start = time.time()
        outputs = attention_seq(batch)
        end = time.time() - start
        attention_times.append(end)
    fourier_times.pop(0)
    attention_times.pop(0)
    fourier_time = sum(fourier_times)/len(fourier_times)
    attention_time = sum(attention_times)/len(attention_times)
    return fourier_time, attention_time


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--seq-len', type=int, help='sequence length')
    ap.add_argument('--embed-len', type=int, help='embedding length')
    ap.add_argument('--batch-size', type=int, help='Batch size')
    args = ap.parse_args()
    seq_lens = [64, 100, 128, 256, 512]#, 1024]
    embed_dims = [32, 64, 128, 256, 512]#, 1024]
    fourier_times = []
    attention_times = []
    output_seq_lens = []
    output_embed_dims = []
    num_heads = 1
    mlp_ratio = 4
    for embed_dim in tqdm(embed_dims, position=0):
        for seq_len in tqdm(seq_lens, position=1):
            #P_Norm = PreNorm
            P_Norm = PostNorm
            attention = P_Norm(embed_dim, Attention(embed_dim, heads=num_heads, dim_head=embed_dim//num_heads, dropout=0))
            fourier = P_Norm(embed_dim, Fourier())
            mlp_attn = P_Norm(embed_dim, FeedForward(embed_dim, mlp_ratio * embed_dim, dropout=0))
            mlp_fourier = P_Norm(embed_dim, FeedForward(embed_dim, mlp_ratio * embed_dim, dropout=0))
            fourier_seq = nn.Sequential(fourier, mlp_fourier)
            attention_seq = nn.Sequential(attention, mlp_attn)

            batch = torch.rand([int(args.batch_size), seq_len, embed_dim])
            device = torch.device('cuda')
            fourier_seq = fourier_seq.to(device)
            attention_seq = attention_seq.to(device)
            batch = batch.to(device)

            fourier_time, attention_time = main(batch, fourier_seq, attention_seq)
            print(f'embed_dim:{embed_dim} | seq_len:{seq_len} | Fourier_time:{fourier_time}, attention_time{attention_time}')
            fourier_times.append(fourier_time)
            attention_times.append(attention_time)
            output_seq_lens.append(seq_len)
            output_embed_dims.append(embed_dim)
    import pandas as pd
    df = pd.DataFrame()
    df['seq_len'] = output_seq_lens
    df['embed_dim'] = output_embed_dims
    df['fourier'] = fourier_times
    df['attention'] = attention_times
    print(df.head())
    df.to_csv("./test_times_1heads_PostNorm.csv", index=False)
