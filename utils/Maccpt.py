from thop import profile
from thop import clever_format
import torch.nn as nn
import torch
from models.network import Mamba_encoder, Mamba_decoder
class net(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.encoder = Mamba_encoder(config)
        self.decoder = Mamba_decoder(config)

    def forward(self,input):

        SNR=20
        x=self.encoder(input, SNR)
        y=self.decoder(x, SNR)
        return y

def test_mem_and_comp(config):
    network=net(config).cuda()
    input=torch.randn(1,3,config.DATA.IMG_SIZE,config.DATA.IMG_SIZE).cuda()
    macs,params=profile(network,inputs=(input,))
    macs, params = clever_format([macs, params], "%.3f")
    torch.cuda.empty_cache()
    del network
    torch.cuda.empty_cache()
    print(macs,params)