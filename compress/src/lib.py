import pandas as pd
import torch

# TODO store wtiles, htiles, tile and shape

def decompress_file(model, fname, shape, batch_size):
    if type(net)==str:
        net = torch.load(model)
        net.eval()
        net.entropy_bottleneck.update()

    collect_out = torch.Tensor()
    f = pd.read_parquet(fname)
    for chunk in range(0, len(f), batch_size):
        chunkdata = f.iloc[chunk:chunk+batch_size]['chunk'].to_list()
        d = net.entropy_bottleneck.decompress(chunkdata, shape)
        o = net.decode(d).detach().cpu()
        collect_out = torch.concat([collect_out, o])
    
    return collect_out