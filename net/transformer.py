import torch
import torch.nn as nn
import torch.nn.functional as F
from net.xcit import XCABlock1D


class ShapeTransformer(nn.Module):
    def __init__(self, in_vtxs, out_vtxs): 
        super(ShapeTransformer, self).__init__()
        in_channels = 3
        out_channels = [16, 32, 64]
        latent_channels = 128
        eta = 1e-5
        # add token vector
        self.enc_token = torch.nn.Parameter(torch.empty(latent_channels), requires_grad=True)
        nn.init.trunc_normal_(self.enc_token, std=0.02)
        self.template_enc = nn.Sequential(
            nn.Linear(in_channels, out_channels[0]),
            torch.nn.ReLU(),
            nn.Linear(out_channels[0], out_channels[1]),
            torch.nn.ReLU(),
            nn.Linear(out_channels[1], out_channels[2]),
            torch.nn.ReLU(),
            nn.Linear(out_channels[2], latent_channels//2),
        )
        self.offset_enc = nn.Sequential(
            nn.Linear(in_channels, out_channels[0], bias=True),
            torch.nn.ReLU(),
            nn.Linear(out_channels[0], out_channels[1], bias=True),
            torch.nn.ReLU(),
            nn.Linear(out_channels[1], out_channels[2], bias=True),
            torch.nn.ReLU(),
            nn.Linear(out_channels[2], latent_channels//2, bias=True),
        )
        self.shape_enc = nn.ModuleList([XCABlock1D(latent_channels, 8, eta=eta), XCABlock1D(latent_channels, 8, eta=eta), XCABlock1D(latent_channels, 8, eta=eta), XCABlock1D(latent_channels, 8, eta=eta)])
        self.shape_dec = nn.ModuleList([XCABlock1D(latent_channels, 8, eta=eta), XCABlock1D(latent_channels, 8, eta=eta), XCABlock1D(latent_channels, 8, eta=eta), XCABlock1D(latent_channels, 8, eta=eta)])
        self.template_dec = nn.Sequential(
            nn.Linear(in_channels, out_channels[0]),
            torch.nn.ReLU(),
            nn.Linear(out_channels[0], out_channels[1]),
            torch.nn.ReLU(),
            nn.Linear(out_channels[1], out_channels[2]),
            torch.nn.ReLU(),
            nn.Linear(out_channels[2], latent_channels),
        )
        self.style_common_mlp = nn.Sequential(
            nn.Linear(latent_channels, latent_channels, bias=True),
            torch.nn.ReLU(),
            nn.Linear(latent_channels, latent_channels, bias=True),
            torch.nn.ReLU(),
            nn.Linear(latent_channels, latent_channels, bias=True),
            torch.nn.ReLU(),
            nn.Linear(latent_channels, latent_channels, bias=True),
            torch.nn.ReLU(),
        ) # StyleMLP(4 common)
        self.style_mlp  = nn.ModuleList([
            nn.Linear(latent_channels, latent_channels, bias=True) for i in range(len(self.shape_dec))
        ]) # and for each layer of xct transformer(as described in sec 3.4)
        self.offset_dec = nn.Sequential(
            nn.Linear(latent_channels, out_channels[2], bias=True),
            torch.nn.ReLU(),
            nn.Linear(out_channels[2], out_channels[1], bias=True),
            torch.nn.ReLU(),
            nn.Linear(out_channels[1], out_channels[0], bias=True),
            torch.nn.ReLU(),
            nn.Linear(out_channels[0], 3, bias=True),
        )

    def encode(self, tx, x):
        z = self.inference_model(tx, x)
        return z

    def decode(self, pt, z):
        logits = self.generative_model(pt, z)
        return logits

    def inference_model(self, tx, x):
        if len(tx.shape) == 2:
            tx = tx[None] # b, v, c
        xt = self.offset_enc(x) # offset token
        pt = torch.broadcast_to(self.template_enc(tx), (x.shape[0], x.shape[1], xt.shape[-1])) # positional token
        x = torch.concat([pt, xt], dim=-1)
        x = torch.concat([x, torch.broadcast_to(self.enc_token, (x.shape[0], 1, x.shape[2]))], dim=1)
        for enc in self.shape_enc:
            x = enc(x)
        
        return x[:, -1]

    def generative_model(self, tx, z):
        x = torch.broadcast_to(self.template_dec(tx[None]), (z.shape[0], tx.shape[0], z.shape[-1])) # positional token
        cz = self.style_common_mlp(z)# common_z
        for smlp, dec in zip(self.style_mlp, self.shape_dec):
            z = torch.broadcast_to(F.relu(smlp(cz))[:, None], x.shape) # style vec
            x = x * z # style modulation
            x = dec(x) # XCiT
        x = self.offset_dec(x) # offset
        return x
    

    def forward(self, tx, x):
        z = self.encode(tx, x)
        y = self.decode(tx, z)
        return y

    def no_weight_decay(self):
        return {'enc_token'}

    def get_tag(self):
        return "shape_transformer"
