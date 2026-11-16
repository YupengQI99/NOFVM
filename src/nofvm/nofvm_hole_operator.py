"""Fixed-hole linear neural operator with a solid-square classical baseline.

The learned field and learned Schur response are coupled by a geometric boundary
projection. True hole response matrices are labels and never model buffers.
"""
import numpy as np
import torch
from torch import nn


def coordinate_features(xy, length=.8):
    x = np.asarray(xy)/length
    result = [x[:, 0],x[:, 1]]
    for axis in (0,1):
        for frequency in range(1,5):
            result += [np.sin(np.pi*frequency*x[:,axis]),np.cos(np.pi*frequency*x[:,axis])]
    return np.array(result).T


class HoleBoundaryOperator(nn.Module):
    def __init__(self, base_h, base_s, xy, B, T):
        super().__init__()
        m = base_s.shape[0]
        i,j = np.triu_indices(m,1)
        weights = -base_s[i,j]
        if np.any(weights<=0) or np.any(base_h<=0):
            raise ValueError("Positive baseline response required")
        boundary_ids = np.flatnonzero(np.asarray(abs(B).sum(axis=1)).ravel()>0)
        boundary_map = B[boundary_ids].toarray().T
        projection = boundary_map.T@np.linalg.inv(boundary_map@boundary_map.T)
        for name,value in (("log_base_h",np.log(base_h)),("log_base_weights",np.log(weights)),
                           ("features",coordinate_features(xy)),("Bboundary",boundary_map),
                           ("T",T),("projection",projection)):
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))
        self.register_buffer("boundary_ids",torch.tensor(boundary_ids,dtype=torch.long))
        self.register_buffer("edge_i",torch.tensor(i,dtype=torch.long))
        self.register_buffer("edge_j",torch.tensor(j,dtype=torch.long))
        self.log_weight_correction=nn.Parameter(torch.zeros(len(i)))
        self.trunk=nn.Sequential(nn.Linear(18,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,m))
        nn.init.zeros_(self.trunk[-1].weight);nn.init.zeros_(self.trunk[-1].bias)

    def raw_field(self,ids=None):
        features=self.features if ids is None else self.features[ids]
        baseline=self.log_base_h if ids is None else self.log_base_h[ids]
        return torch.softmax(baseline+self.trunk(features),dim=1)

    def schur(self):
        logw=self.log_base_weights+self.log_weight_correction
        weights=logw.exp()
        s=weights.new_zeros(self.T.shape)
        s[self.edge_i,self.edge_j]=-weights
        s[self.edge_j,self.edge_i]=-weights
        return s-torch.diag(s.sum(dim=1)),logw

    def boundary_field(self,s):
        raw=self.raw_field(self.boundary_ids)
        return raw+self.projection@(self.T-s-self.Bboundary@raw)


def deploy_response(model):
    with torch.no_grad():
        raw=model.raw_field().cpu().numpy().astype(float)
        _,logs=model.schur()
        weights=logs.cpu().numpy().astype(float)
        weights=np.exp(weights)
        i,j=model.edge_i.cpu().numpy(),model.edge_j.cpu().numpy()
        m=model.T.shape[0]
        s=np.zeros((m,m));s[i,j]=s[j,i]=-weights
        s-=np.diag(s.sum(axis=1))
        raw/=raw.sum(axis=1,keepdims=True)
        B=model.Bboundary.cpu().numpy().astype(float)
        T=model.T.cpu().numpy().astype(float)
        ids=model.boundary_ids.cpu().numpy()
    # Recompute projection in float64; avoid precision loss of saved float32 inverse.
    correction=B.T@np.linalg.solve(B@B.T,T-s-B@raw[ids])
    raw[ids]+=correction
    return dict(H=raw,S=s)
