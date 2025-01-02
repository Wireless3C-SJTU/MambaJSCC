'''
@author: Tong Wu
@contact: wu_tong@sjtu.edu.cn
'''

from models.network import Mamba_encoder, Mamba_decoder
from models.channel import Channel
from data.datasets import get_loader
import torch
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from utils.utils import *
from utils.distortion import *
import time


def eval_MambaJSCC(config):
    _, test_loader = get_loader(config)

    encoder_path = config.TRAIN.ENCODER_PATH + "OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(config.MODEL.VSSM.OUT_CHANS,config.MODEL.VSSM.Extent,config.TRAIN.LOSS,config.MODEL.VSSM.SCAN_NUMBER,config.CHANNEL.SNR,config.CHANNEL.ADAPTIVE, config.CHANNEL.TYPE,len(config.MODEL.VSSM.EMBED_DIM), config.MODEL.VSSM.EMBED_DIM,config.MODEL.VSSM.DEPTHS,config.DATA.IMG_SIZE) + '.pt'
    decoder_path = config.TRAIN.DECODER_PATH + "OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(config.MODEL.VSSM.OUT_CHANS,config.MODEL.VSSM.Extent,config.TRAIN.LOSS,config.MODEL.VSSM.SCAN_NUMBER,config.CHANNEL.SNR,config.CHANNEL.ADAPTIVE, config.CHANNEL.TYPE,len(config.MODEL.VSSM.EMBED_DIM), config.MODEL.VSSM.EMBED_DIM,config.MODEL.VSSM.DEPTHS,config.DATA.IMG_SIZE) + '.pt'

    encoder=torch.load(encoder_path)
    decoder=torch.load(decoder_path)

    channel=Channel(config)
    B,C,H,W=next(iter(test_loader))[0].shape
    test_mem_and_comp(config, encoder, decoder, input_size=(H,W))
    
    print(H,W)
    matrix=eval_matrix(config) 
    encoder.eval()
    decoder.eval()
    performance_all=[] 
    #SNR_list = [20] #config.CHANNEL.SNR
    SNR_list=config.CHANNEL.SNR
    print("----------Evaluating: ls:128--OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(config.MODEL.VSSM.OUT_CHANS,config.MODEL.VSSM.Extent,config.TRAIN.LOSS,config.MODEL.VSSM.SCAN_NUMBER,config.CHANNEL.SNR,config.CHANNEL.ADAPTIVE, config.CHANNEL.TYPE,len(config.MODEL.VSSM.EMBED_DIM), config.MODEL.VSSM.EMBED_DIM,config.MODEL.VSSM.DEPTHS,config.DATA.IMG_SIZE))
    all_time=0
    
    for SNR in SNR_list:
        
        number=0
        performance_avg=0
        seed_torch()
        with tqdm(test_loader, dynamic_ncols=False) as tqdmTestData:
            for i, (input_image, target) in enumerate(tqdmTestData):
                input_image = input_image.cuda()
                if config.DATA.DATASET=='CIFAR10':
                    input_image=torch.nn.functional.interpolate(input_image, (128,128),mode='nearest')
               # print(input_image.shape)
                #target = target.cuda()
                start_encoder=time.time()
                feature = encoder(input_image, SNR)
                end_encoder=time.time()
                CBR=feature.numel()/2/input_image.numel()
                

                received, pwr, h = channel.forward(feature, SNR)
                if config.CHANNEL.TYPE=='rayleigh':
                    sigma_square = 1.0 / (10 ** (SNR / 10))
                    received = torch.conj(h) * received / (torch.abs(h) ** 2 + sigma_square)
                    
                elif config.CHANNEL.TYPE=='awgn':
                    pass
                else:
                    raise ValueError("channel type error")

                received = torch.cat((torch.real(received), torch.imag(received)), dim=2) * torch.sqrt(pwr)
                start_decoder=time.time()
                recon_image = decoder(received, SNR)
                end_decocer=time.time()
                all_time=all_time+end_encoder-start_encoder+end_decocer-start_decoder

                performance=matrix(recon_image, input_image)
                performance_avg=performance_avg+performance
                tqdmTestData.set_postfix({
                    'matrix':performance,
                    'CBR':CBR,
                    'SNR':SNR,
                    'per':(performance,performance_avg/(i+1))
                }
                    )

        performance_all.append(performance_avg/(i+1))
        
    print(all_time/(len(SNR_list)*len(test_loader)*config.DATA.TEST_BATCH))
    print("SNRs:", SNR_list)
    print("performance:", performance_all)



def eval_MambaJSCC_with_SNR_error(config,mode=2):
    '''
    SNR error with Gaussian random distribution
    mode 1 stand for fix estimation with various SNR
    mode 2 stand for fix SNR with various estimation
    '''
    _, test_loader = get_loader(config)

    encoder_path = config.TRAIN.ENCODER_PATH + "OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(config.MODEL.VSSM.OUT_CHANS,config.MODEL.VSSM.Extent,config.TRAIN.LOSS,config.MODEL.VSSM.SCAN_NUMBER,config.CHANNEL.SNR,config.CHANNEL.ADAPTIVE, config.CHANNEL.TYPE,len(config.MODEL.VSSM.EMBED_DIM), config.MODEL.VSSM.EMBED_DIM,config.MODEL.VSSM.DEPTHS,config.DATA.IMG_SIZE) + '.pt'
    decoder_path = config.TRAIN.DECODER_PATH + "OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(config.MODEL.VSSM.OUT_CHANS,config.MODEL.VSSM.Extent,config.TRAIN.LOSS,config.MODEL.VSSM.SCAN_NUMBER,config.CHANNEL.SNR,config.CHANNEL.ADAPTIVE, config.CHANNEL.TYPE,len(config.MODEL.VSSM.EMBED_DIM), config.MODEL.VSSM.EMBED_DIM,config.MODEL.VSSM.DEPTHS,config.DATA.IMG_SIZE) + '.pt'

    encoder=torch.load(encoder_path)
    decoder=torch.load(decoder_path)

    channel=Channel(config)

    matrix=eval_matrix(config) 
    encoder.eval()
    decoder.eval()
     

    SNR_list=[1,5,10,15,20]#config.CHANNEL.SNR
    error_rate=[0.01, 0.1, 0.5, 1, 2]
    print("----------Evaluating SNR error :OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(config.MODEL.VSSM.OUT_CHANS,config.MODEL.VSSM.Extent,config.TRAIN.LOSS,config.MODEL.VSSM.SCAN_NUMBER,config.CHANNEL.SNR,config.CHANNEL.ADAPTIVE, config.CHANNEL.TYPE,len(config.MODEL.VSSM.EMBED_DIM), config.MODEL.VSSM.EMBED_DIM,config.MODEL.VSSM.DEPTHS,config.DATA.IMG_SIZE))
    for error in error_rate:
        performance_all=[]
        for SNR in SNR_list:
            number=0
            performance_avg=0
            seed_torch()
            with tqdm(test_loader, dynamic_ncols=False) as tqdmTestData:
                for i, (input_image, target) in enumerate(tqdmTestData):
                    input_image = input_image.cuda()
                    SNR_error=SNR+np.random.normal(0, error)

                    if mode==1:
                        feature = encoder(input_image, SNR)
                        received, pwr, h = channel.forward(feature, SNR_error)
                        if config.CHANNEL.TYPE=='rayleigh':
                            sigma_square = 1.0 / (10 ** (SNR / 10))
                            received = torch.conj(h) * received / (torch.abs(h) ** 2 + sigma_square)
                            #print(1)
                        elif config.CHANNEL.TYPE=='awgn':
                            pass
                        else:
                            raise ValueError("channel type error")

                        received = torch.cat((torch.real(received), torch.imag(received)), dim=2) * torch.sqrt(pwr)
                        
                        recon_image = decoder(received, SNR)

                    elif mode==2:
                        feature = encoder(input_image, SNR_error)
                        received, pwr, h = channel.forward(feature, SNR)
                        if config.CHANNEL.TYPE=='rayleigh':
                            sigma_square = 1.0 / (10 ** (SNR_error / 10))
                            received = torch.conj(h) * received / (torch.abs(h) ** 2 + sigma_square)
                            #print(1)
                        elif config.CHANNEL.TYPE=='awgn':
                            pass
                        else:
                            raise ValueError("channel type error")

                        received = torch.cat((torch.real(received), torch.imag(received)), dim=2) * torch.sqrt(pwr)
                        
                        recon_image = decoder(received, SNR_error)
                    
                    CBR=feature.numel()/2/input_image.numel()
                    performance=matrix(recon_image, input_image)
                    performance_avg=performance_avg+performance
                    tqdmTestData.set_postfix({
                        'matrix':performance,
                        'CBR':CBR,
                        'SNR':SNR,
                        'SNR_error':SNR_error,
                        'per':(performance,performance_avg/(i+1))
                    }
                        )

            performance_all.append(performance_avg/(i+1))
            
        
        print("SNRs:", SNR_list)
        print(f"performance with {error}:", performance_all)


def test_mem_and_comp(config,encoder,decoder,input_size=(256,256)):
    from torch_operation_counter import OperationsCounterMode
    class net(torch.nn.Module):

        def __init__(self,encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self,input):

            SNR=20
            x=self.encoder(input, SNR)
            y=self.decoder(x, SNR)
            return y
    
    network=net(encoder,decoder).cuda()
    input=torch.randn(1,3,input_size[0],input_size[1]).cuda()
    with OperationsCounterMode(network) as ops_counter:
        network(input)
    #macs,params=profile(network,inputs=(input,))
    #macs, params = clever_format([macs, params], "%.5f")
    print("MACs:{}G. Paras:{}M.".format(ops_counter.total_operations/1e9, sum([p.numel() for p in [*network.parameters()][:-1]]) / 1e6,))
