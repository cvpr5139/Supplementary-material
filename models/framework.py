from __future__ import absolute_import, print_function
from torch import nn
from models.ITAE import ITAE_encoder, ITAE_decoder, ITAE_encoder_x4, ITAE_decoder_x4

class AE(nn.Module):
    def __init__(self, chnum_in, model, one_path = False):
        super(AE, self).__init__()
        self.chnum_in = chnum_in
        if model == 'ITAE':
            self.encoder = ITAE_encoder(self.chnum_in, one_path)
            self.decoder = ITAE_decoder(self.chnum_in, one_path)
        elif model =='ITAE_ped':
            self.encoder = ITAE_encoder_x4(self.chnum_in, one_path)
            self.decoder = ITAE_decoder_x4(self.chnum_in, one_path)
        else:
            print('Cannot initial model')
    def forward(self, x):

        z, flow_input = self.encoder(x)
        out = self.decoder(z)

        return out, flow_input