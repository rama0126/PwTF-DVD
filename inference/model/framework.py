
import torch
import torch.nn as nn
from .video_encoder import I3D8x8
from .attention_network import APNResNet
from .transformers import TransformerHead, SpatialTransformerE


# Model for deployment
# Note: modified model structure for easier deployment combined with video encoder FTCN official repo code
def get_model():
    part_num = 5
    model= I3D8x8()
    model_ft = APNResNet(partials_num=part_num,depth =50)
    params = dict(spatial_size=14, time_size=16, in_channels=1024,num_parts=part_num)
    TTE = TransformerHead(**params)
    STE = SpatialTransformerE(**params)
    MLP = torch.nn.Linear(2048,1)
    return Framework(model, model_ft, TTE, STE, MLP)
    
    
class Framework(nn.Module):
    def __init__(self, model, model_ft, TTE, STE, MLP):
        super(Framework, self).__init__()
        self.model = model
        self.model_ft = model_ft
        self.TTE = TTE
        self.STE = STE
        self.MLP = MLP
        
    def forward(self, video_sample, ft_sample):
        out1, out2, out3, out4, out5,out6,  (xs, scaled_x) = self.model_ft(ft_sample.float())
        ft_feats = [out1, out2, out3, out4, None]
        x,_ = self.model(video_sample,ft_feats)
        ft_s , _= self.STE(x,out5,xs)
        ft_t, _ = self.TTE(x,out6)
        outputs = self.MLP(torch.concat((ft_t, ft_s), dim = 1))
        return outputs
