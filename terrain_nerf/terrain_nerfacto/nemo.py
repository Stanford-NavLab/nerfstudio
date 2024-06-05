import tinycudann as tcnn
import torch
import torch.nn as nn
from torch import autograd


class Nemo(nn.Module):
    def __init__(self, spatial_distortion, hash=True) -> None:
        super().__init__()
        
        self.spatial_distortion = spatial_distortion

        self.encoder = tcnn.Encoding(
            n_input_dims=2, 
            encoding_config={
                "otype": "HashGrid" if hash else "DenseGrid",
                "n_levels": 8,
                "n_features_per_level": 8,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.2599210739135742,
                "interpolation": "Smoothstep"
            }
        )
        
        n_neurons = 256
        n_hidden_layers = 3
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.n_output_dims, n_neurons),
            nn.ReLU(True),
            nn.Linear(n_neurons, n_neurons),
            nn.ReLU(True),
            nn.Linear(n_neurons, 1)
        )
        # self.decoder = tcnn.Network(
        #     n_input_dims=self.encoder.n_output_dims,
        #     n_output_dims=1,
        #     network_config={
        #         "otype": "CutlassMLP",
        #         "activation": "ReLU",   
        #         "output_activation": "None",
        #         "n_neurons": 256,
        #         "n_hidden_layers": 3,
        #     },
        # )

    def forward(self, x):
        inp_shape = x.shape
        # x = self.spatial_distortion(x)  
        # x = (x + 2.0) / 4.0
        encoded = self.encoder(x.view(-1, 2)).to(dtype=torch.float)
        z = self.decoder(encoded)
        return z.view(*inp_shape[:-1], -1)

    def forward_with_grad(self, x):
        with torch.enable_grad():
            x = x.requires_grad_(True)
            z = self.forward(x)
            grad = autograd.grad(
                z,
                x,
                torch.ones_like(z, device=x.device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        return z, grad