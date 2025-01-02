'''
@author: Tong Wu
@contact: wu_tong@sjtu.edu.cn
'''
from models.encoder import create_encoder
from models.decoder import create_decoder
import torch.nn as nn


class Mamba_encoder(nn.Module):
    def __init__(self, config):
        super(Mamba_encoder, self).__init__()
        self.config = config
        self.encoder = create_encoder(config)


    def forward(self, input_image, SNR):
        feature = self.encoder(input_image, SNR)
        
        return feature

class Mamba_decoder(nn.Module):
    def __init__(self, config):
        super(Mamba_decoder, self).__init__()
        self.config = config

        self.decoder = create_decoder(config)


    def forward(self, feature, SNR):
        recon_image = self.decoder(feature, SNR)

        return recon_image
    
class Mamba_classify(nn.Module):
    def __init__(self, config):
        super(Mamba_decoder, self).__init__()
        self.config = config

        self.decoder = create_decoder(config)


    def forward(self, feature, SNR):
        recon_image = self.decoder(feature, SNR)

        return recon_image