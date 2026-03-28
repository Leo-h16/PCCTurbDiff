import torch
import torch.nn as nn
import os
import sys
import diffusion_net

class MeshEncoder(nn.Module):
    def __init__(self, C_in=3, C_out=256, C_width=256, N_block=4):
        super().__init__()
        self.diffusion_net = diffusion_net.layers.DiffusionNet(
            C_in=C_in, 
            C_out=C_out, 
            C_width=C_width, 
            N_block=N_block, 
            outputs_at='vertices'
        )

    def load_from_checkpoint(self, checkpoint_path, device='cpu'):
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
                
            state_dict = torch.load(checkpoint_path, map_location=device)
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('encoder.'):
                    name = k.replace('encoder.', '')
                    new_state_dict[name] = v
            
            self.diffusion_net.load_state_dict(new_state_dict, strict=True)
            for param in self.parameters():
                param.requires_grad = False
            self.eval() 
            print(f"Successfully loaded MeshEncoder (DiffusionNet) weights from {checkpoint_path}")

    def forward(self, verts, mass, L, evals, evecs, gradX, gradY, faces, **kwargs):
        device = next(self.parameters()).device

        verts = verts.to(device)
        mass = mass.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        faces = faces.to(device)

        if L.is_sparse:
            L = L.coalesce().clone().to(device)
        else:
            L = L.clone().to(device)

        if gradX.is_sparse:
            gradX = gradX.coalesce().clone().to(device)
        else:
            gradX = gradX.clone().to(device)

        if gradY.is_sparse:
            gradY = gradY.coalesce().clone().to(device)
        else:
            gradY = gradY.clone().to(device)

        x_local = self.diffusion_net(
            verts, mass, L=L, evals=evals, evecs=evecs,
            gradX=gradX, gradY=gradY, faces=faces
        )

        if x_local.dim() == 3:
            global_features = x_local.mean(dim=1)
        else:
            global_features = x_local.mean(dim=0, keepdim=True)

        return global_features