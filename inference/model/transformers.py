
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
# Removed Rearrange import - not used in inference
# Removed unused imports for inference
from math import sqrt
from config_ftcn import config as my_cfg

# Initialize config
my_cfg.init_with_yaml()
my_cfg.update_with_yaml("ftcn_tt.yaml")
my_cfg.freeze()

# =============================================================================
# Basic Transformer Components
# =============================================================================

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask
        
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

# =============================================================================
# Utility Functions
# =============================================================================

def valid_idx(idx, h):
    i = idx // h
    j = idx % h
    if j == 0 or i == h - 1 or j == h - 1:
        return False
    else:
        return True


# =============================================================================
# Patch Pooling Classes (Simplified for Inference)
# =============================================================================

class CenterPatchPool(nn.Module):
    """Simplified patch pooling for inference - always use center patch"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # batch,channel,16,7x7
        b, c, t, h, w = x.shape
        x = x.reshape(b, c, t, h * w)
        idx = h * w // 2  # Always use center patch for inference
        x = x[..., idx]
        return x

class CenterAvgPool(nn.Module):
    """Simplified average pooling for inference - use all valid patches"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # batch,channel,16,7x7
        b, c, t, h, w = x.shape
        x = x.reshape(b, c, t, h * w)
        candidates = list(range(h * w))
        candidates = [idx for idx in candidates if valid_idx(idx, h)]
        x = x[..., candidates].mean(-1)
        return x

class CenterSelect(nn.Module):
    """Simplified selection for inference - use all valid patches"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # batch,7x7
        size = x.shape[1]
        h = int(sqrt(size))
        candidates = list(range(size))
        candidates = [idx for idx in candidates if valid_idx(idx, h)]
        x = x[:, candidates]
        return x

# =============================================================================
# Vision Transformer Classes
# =============================================================================

# Removed ViT and VideoiT classes - not used in inference

# =============================================================================
# Specialized Transformer Classes
# =============================================================================

# SpatialTransformer class - used by SpatialTransformerE
class SpatialTransformer(nn.Module):
    def __init__(self,num_patches, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.dim =dim
        self.num_patches=num_patches
        self.pos_embedding = nn.Parameter(posemb_sincos_2d(14,14,dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.fre_pos = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(1024, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(1024), nn.Linear(1024, 1024)
        )
        self.freq_embedding = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim))
        self.cls_token_pos = nn.Parameter(torch.randn(1, 1, dim))
        self.mlp_head_ste = nn.Linear(dim, 1)
        
    def posemb_loc(self, locs,batch_size):
        if locs is not None:
            b,n,_ = locs.shape

        pos224 =self.pos_embedding.view(14,14,-1)
        pos_locs = torch.zeros(batch_size,n, self.dim)
        if locs is not None:
            # locs[locs>=1] = 0.999999
            loc_x = torch.floor(locs[:,:,0]*6)
            loc_y = torch.floor(locs[:,:,1]*6)

            loc_x = loc_x.long()
            loc_y = loc_y.long()
            x_a = locs[:,:,0]*6 - loc_x
            y_a = locs[:,:,0]*6 - loc_y
            
        x1_a = torch.sqrt( torch.pow(1-x_a,2) + torch.pow(1-y_a,2))
        x2_a = torch.sqrt( torch.pow(1-x_a,2) + torch.pow(y_a,2))
        x3_a = torch.sqrt( torch.pow(x_a,2) + torch.pow(1-y_a,2))
        x4_a = torch.sqrt( torch.pow(x_a,2) + torch.pow(y_a,2))
        pos_locs[:, :, :] = pos224[loc_x,loc_y,:]*x1_a.unsqueeze(-1)\
                            +pos224[loc_x+1,loc_y,:]*x2_a.unsqueeze(-1)\
                            +pos224[loc_x,loc_y+1,:]*x3_a.unsqueeze(-1)\
                            +pos224[loc_x+1,loc_y+1,:]*x4_a.unsqueeze(-1)
        
        return pos_locs
        
    def forward(self, x,ft_feature,locs):
        b, n, _ = x.shape #batch,num_patches,channels  #
        cls_tokens_pos = repeat(self.cls_token_pos, '() n d -> b n d', b = b)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        cls_tokens = cls_tokens+cls_tokens_pos
        x += self.pos_embedding
        if locs is not None:
            posemb_loc = self.posemb_loc(locs,b).clone().detach()
            if torch.cuda.is_available() and x.is_cuda:
                posemb_loc = posemb_loc.cuda()
            posemb_loc += self.fre_pos

            ft_feature = ft_feature+posemb_loc
            ft_feature = self.freq_embedding(ft_feature)
            x = torch.cat((cls_tokens, x,ft_feature), dim=1)
        else :
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        x = self.transformer(x, mask=None)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        
        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x, self.mlp_head_ste(x.clone())
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    assert (dim % 4) == 0, "dim must be multiple of 4"
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

class SpatialTransformer(nn.Module):
    def __init__(self, num_patches, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.register_buffer('pos_embedding', posemb_sincos_2d(14, 14, dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token_pos = nn.Parameter(torch.randn(1, 1, dim))
        self.fre_pos = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.spatial_proj = nn.Linear(dim, dim)
        self.freq_embedding = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # output ES ∈ R^1024
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1024))
        self.mlp_head_ste = nn.Linear(dim, 1)

    @torch.no_grad()
    def posemb_loc(self, locs, batch_size):
        if locs is None:
            return None
        # locs: (B, P, 2) in [0,1]
        B, P, _ = locs.shape
        device, dtype = locs.device, locs.dtype

        pos_grid = self.pos_embedding.view(14, 14, self.dim).to(device)  # (14,14,dim) - already [196, dim]

        S = 13.0
        x_f = (locs[:, :, 0] * S).clamp(0, 13 - 1e-6)
        y_f = (locs[:, :, 1] * S).clamp(0, 13 - 1e-6)

        ix = torch.floor(x_f).long().clamp(0, 12)
        iy = torch.floor(y_f).long().clamp(0, 12)
        ix1 = (ix + 1).clamp(max=13)
        iy1 = (iy + 1).clamp(max=13)

        x_a = (x_f - ix).to(dtype)  # frac
        y_a = (y_f - iy).to(dtype)

        w11 = (1 - x_a) * (1 - y_a)
        w12 = x_a * (1 - y_a)
        w21 = (1 - x_a) * y_a
        w22 = x_a * y_a

        p11 = pos_grid[ix,  iy,  :]  # (B,P,dim) via advanced indexing
        p12 = pos_grid[ix1, iy,  :]
        p21 = pos_grid[ix,  iy1, :]
        p22 = pos_grid[ix1, iy1, :]

        pos_locs = (p11 * w11.unsqueeze(-1) +
                    p12 * w12.unsqueeze(-1) +
                    p21 * w21.unsqueeze(-1) +
                    p22 * w22.unsqueeze(-1))  # (B,P,dim)
        return pos_locs

    def forward(self, x, ft_feature, locs):
        # x: (B, N=14*14, dim), ft_feature: (B, P, dim), locs: (B, P, 2)
        B, N, _ = x.shape

        # cls token (+ pos)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=B)
        cls_tokens_pos = repeat(self.cls_token_pos, '() n d -> b n d', b=B)
        cls_tokens = cls_tokens + cls_tokens_pos

        # Wsp z_sp + pos_sp
        x = self.spatial_proj(x)
        x = x + self.pos_embedding.unsqueeze(0).to(x.device)

        # part tokens: Wfreq Z_p + (interp(pos_sp) + pos_freq)
        if locs is not None and ft_feature is not None:
            pospart = self.posemb_loc(locs, B)  # (B,P,dim)
            if pospart is not None:
                pospart = pospart + self.fre_pos  # + posfreq
            ft_feature = self.freq_embedding(ft_feature)  # Wfreq
            if pospart is not None:
                ft_feature = ft_feature + pospart
            tokens = torch.cat((cls_tokens, x, ft_feature), dim=1)
        else:
            tokens = torch.cat((cls_tokens, x), dim=1)

        tokens = self.dropout(tokens)
        tokens = self.transformer(tokens, mask=None)
        out = tokens.mean(dim=1) if self.pool == 'mean' else tokens[:, 0]

        out = self.to_latent(out)
        es = self.mlp_head(out)            # ES ∈ R^1024
        ste_score = self.mlp_head_ste(out) # optional head

        return es, ste_score

class TimeTransformer(nn.Module):
    def __init__(self,num_patches, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.num_patches=num_patches
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.freq_embedding = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim))
        self.pool = pool
        self.to_latent = nn.Identity()
        self.LN = nn.LayerNorm(dim)
        
    def forward(self, x,ft_t):
        b, n, _ = x.shape #batch,num_patches,channels  #
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)

        ft_t = self.freq_embedding(ft_t)
        x +=ft_t.unsqueeze(1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
  
        # x = torch.cat((x,ft_t), dim=1)
        x = self.dropout(x)
        x = self.transformer(x, mask=None)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        x = self.LN(x)
       
        return x, None

# =============================================================================
# FTCN-specific Transformer Classes
# =============================================================================

class SpatialTransformerE(nn.Module):
    def __init__(self, spatial_size=14, time_size = 16, in_channels = 1024, num_parts=5):
        super().__init__()
        self.num_parts = num_parts
        self.in_channels = in_channels
        default_params= dict(
            dim=self.in_channels, depth=1, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1,
            num_patches = spatial_size ** 2, num_classes = 1
        )
        self.num_patches = spatial_size ** 2
        self.freq_embedding = nn.Linear(2048,1024)
        self.freq_embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.freq_embedding.bias.data.zero_()
        self.pool =  nn.AvgPool3d((time_size, 1, 1))
        self.spatial_T = SpatialTransformer( **default_params )
        
    def forward(self, x, ft, locs):
        batch_size = x.shape[0]
        x = self.pool(x)
        if self.num_parts > 0:
            ft = self.freq_embedding(ft.reshape(batch_size*self.num_parts,2048))
            ft = ft.view(batch_size,self.num_parts,1024)
            locs = locs.reshape(-1, self.num_parts, 2)
        x = x.view(batch_size,self.num_patches,1024)    
        x = self.spatial_T(x,ft,locs)
        return x    

class TransformerHead(nn.Module):
    def __init__(self, spatial_size=7, time_size=16, in_channels=1024, num_parts=5):
        super().__init__()
        if my_cfg.model.inco.no_time_pool:
            time_size = time_size * 2
        patch_type = my_cfg.model.transformer.patch_type # time
        if patch_type == "time":
            self.pool = nn.AvgPool3d((1, spatial_size, spatial_size))
            self.num_patches = time_size
        elif patch_type == "spatial":
            self.pool = nn.AvgPool3d((time_size, 1, 1))
            self.num_patches = spatial_size ** 2
        elif patch_type == "random":
            self.pool = CenterPatchPool()
            self.num_patches = time_size
        elif patch_type == "random_avg":
            self.pool = CenterAvgPool()
            self.num_patches = time_size
        elif patch_type == "all":
            self.pool = nn.Identity()
            self.num_patches = time_size * spatial_size * spatial_size
        else:
            raise NotImplementedError(patch_type)

        self.dim = my_cfg.model.transformer.dim # False
        if self.dim == -1:
            self.dim = in_channels # 2048
            my_cfg.model.transformer.dim = self.dim

        self.in_channels = in_channels

        if self.dim != self.in_channels:
            self.fc = nn.Linear(self.in_channels, self.dim)
        
        default_params = dict(
            dim=self.dim, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1,
        )
        params = my_cfg.model.transformer.to_dict()
        for key in default_params:
            if key in params:
                default_params[key] = params[key]
        
        self.time_T = TimeTransformer(
            num_patches=self.num_patches, num_classes=1, **default_params
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,ft_t):
        x = self.pool(x)
        x = x.reshape(-1, self.in_channels, self.num_patches)
        x = x.permute(0, 2, 1)
        if self.dim != self.in_channels:
            x = self.fc(x.reshape(-1, self.in_channels))
            x = x.reshape(-1, self.num_patches, self.dim)
        
        x = self.time_T(x,ft_t)
        return x         