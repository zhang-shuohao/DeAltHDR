import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
import math
from importlib import import_module
import numpy as np


def make_model(opt):
    """Create DeAltHDR model with dual encoder and mixed training support"""
    model_config = {
        'inp_channels': opt['n_colors'],
        'out_channels': opt['n_colors'],
        'dim': opt['dim'],
        'Enc_blocks': opt['Enc_blocks'],
        'Middle_blocks': opt['Middle_blocks'],
        'Dec_blocks': opt['Dec_blocks'],
        'num_refinement_blocks': opt.get('num_refinement_blocks', 1),
        'ffn_expansion_factor': opt.get('ffn_expansion_factor', 1),
        'bias': opt.get('bias', False),
        'LayerNorm_type': opt.get('LayerNorm_type', 'WithBias'),
        'num_heads_blks': opt.get('num_heads_blks', [1,2,4,8]),
        'encoder1_attn_type1': opt['encoder1_attn_type1'],
        'encoder1_attn_type2': opt['encoder1_attn_type2'],
        'encoder2_attn_type1': opt['encoder2_attn_type1'],
        'encoder2_attn_type2': opt['encoder2_attn_type2'],
        'encoder3_attn_type1': opt['encoder3_attn_type1'],
        'encoder3_attn_type2': opt['encoder3_attn_type2'],
        'decoder1_attn_type1': opt['decoder1_attn_type1'],
        'decoder1_attn_type2': opt['decoder1_attn_type2'],
        'decoder2_attn_type1': opt['decoder2_attn_type1'],
        'decoder2_attn_type2': opt['decoder2_attn_type2'],
        'decoder3_attn_type1': opt['decoder3_attn_type1'],
        'decoder3_attn_type2': opt['decoder3_attn_type2'],
        'encoder1_ffw_type': opt['encoder1_ffw_type'],
        'encoder2_ffw_type': opt['encoder2_ffw_type'],
        'encoder3_ffw_type': opt['encoder3_ffw_type'],
        'decoder1_ffw_type': opt['decoder1_ffw_type'],
        'decoder2_ffw_type': opt['decoder2_ffw_type'],
        'decoder3_ffw_type': opt['decoder3_ffw_type'],
        'latent_attn_type1': opt['latent_attn_type1'],
        'latent_attn_type2': opt['latent_attn_type2'],
        'latent_attn_type3': opt['latent_attn_type3'],
        'latent_ffw_type': opt['latent_ffw_type'],
        'refinement_attn_type1': opt['refinement_attn_type1'],
        'refinement_attn_type2': opt['refinement_attn_type2'],
        'refinement_ffw_type': opt['refinement_ffw_type'],
        'use_both_input': opt['use_both_input'],
        'num_frames_tocache': opt.get('num_frames_tocache', 4),
        'use_dual_encoder': opt.get('use_dual_encoder', True),
        'training_mode': opt.get('training_mode', 'mixed'),
        'sensitivity': opt.get('sensitivity', 15.0)  # Default sensitivity parameter
    }
    return DeAltHDR(**model_config)


def create_video_model(opt):
    module = import_module('basicsr.models.archs.dealthdr_arch')
    model = module.make_model(opt)
    return model

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def warp(x, flow):
    """Warp image x according to flow field"""
    B, C, H, W = x.size()
    # Create coordinate grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    
    if x.is_cuda:
        grid = grid.cuda()
    
    vgrid = grid + flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='border')
    return output

def compute_occlusion_mask(flow_fw, flow_bw, threshold=0.5):
    """Compute occlusion mask based on forward-backward consistency"""
    # Warp forward flow with backward flow
    flow_fw_warped = warp(flow_fw, flow_bw)
    # Compute consistency
    flow_diff = torch.norm(flow_fw + flow_fw_warped, dim=1, keepdim=True)
    occlusion = (flow_diff > threshold).float()
    return occlusion

class SPyNet(nn.Module):
    """SPyNet optical flow estimation network"""
    def __init__(self):
        super(SPyNet, self).__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(6, 32, 7, 2, 3)
        self.conv2 = nn.Conv2d(32, 64, 7, 2, 3)
        self.conv3 = nn.Conv2d(64, 96, 5, 2, 2)
        self.conv4 = nn.Conv2d(96, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 192, 3, 2, 1)
        
        # Flow prediction
        self.flow_pred = nn.Conv2d(192, 2, 3, 1, 1)
        
    def forward(self, img1, img2):
        """Estimate optical flow from img1 to img2"""
        x = torch.cat([img1, img2], dim=1)
        
        # Feature extraction
        f1 = F.relu(self.conv1(x))
        f2 = F.relu(self.conv2(f1))
        f3 = F.relu(self.conv3(f2))
        f4 = F.relu(self.conv4(f3))
        f5 = F.relu(self.conv5(f4))
        
        # Flow prediction
        flow = self.flow_pred(f5)
        
        # Upsample to original resolution
        flow = F.interpolate(flow, size=img1.shape[2:], mode='bilinear', align_corners=False)
        
        return flow

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

def clipped_softmax(tensor, dim=-1):
    # Create a mask for zero elements
    zero_mask = tensor == 0
    
    # Apply the mask to ignore zero elements in the softmax computation
    # Set zero elements to `-inf` so that they become 0 after softmax
    masked_tensor = tensor.masked_fill(zero_mask, float('-inf'))
    
    # Compute softmax on the modified tensor
    softmaxed = F.softmax(masked_tensor, dim=dim)
    
    # Zero out `-inf` elements (which are now 0 due to softmax) if any original zeros existed
    softmaxed = softmaxed.masked_fill(zero_mask, 0)
    
    non_zero_softmaxed_sum = softmaxed.sum(dim=dim, keepdim=True)
    normalized_softmaxed = softmaxed / non_zero_softmaxed_sum
    
    return normalized_softmaxed

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
# Gated Feed-Forward Network
class GatedFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GatedFeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, 
                                kernel_size=3, stride=1, 
                                padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

# Feed_Forward Network
class FeedForward(nn.Module):
    def __init__(self, c, FFN_Expand=2, drop_out_rate=0.):
        super(FeedForward, self).__init__()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, 
                               out_channels=ffn_channel, 
                               kernel_size=1, 
                               padding=0, 
                               stride=1, 
                               groups=1, 
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, 
                               out_channels=c, 
                               kernel_size=1, 
                               padding=0, 
                               stride=1, 
                               groups=1, 
                               bias=True)
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.conv4(inp)
        x = F.gelu(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return x * self.gamma

##########################################################################
## History based Attentions

class FrameHistoryRouter(nn.Module):
    def __init__(self, dim, num_heads, bias, num_frames_tocache=4):
        """
        Initializes the FrameHistoryRouter module for T-2,T-1,T+1,T+2 frame caching.
        """
        super(FrameHistoryRouter, self).__init__()
        self.dim = dim
        self.bias = bias
        self.num_heads= num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.num_frames_tocache = num_frames_tocache
        
    def forward(self, x, k_cached=None, v_cached=None):
        """
        Forward pass of the FrameHistoryRouter.
        Given the history states (T-2,T-1,T+1,T+2), it aggregates critical features for the restoration of the input frame
        """
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Concatenate cached key and value tensors if provided
        if k_cached is not None and v_cached is not None:
            k = torch.cat([k_cached, k], dim=2)
            v = torch.cat([v_cached, v], dim=2)
        
        # Calculating Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        num_cache_to_keep = int(self.num_frames_tocache*c/self.num_heads)
        return out, k[:, :, -num_cache_to_keep:, :], v[:, :, -num_cache_to_keep:, :]

class FlowGuidedMaskedAttention(nn.Module):
    """
    Flow-Guided Masked Attention (FGMA) as described in DeAltHDR paper.
    Uses optical flow for reliable regions and sparse attention for unreliable regions.
    """
    def __init__(self, dim, num_heads, bias, num_frames_tocache=4, sensitivity=15.0):
        super(FlowGuidedMaskedAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.bias = bias
        self.num_frames_tocache = num_frames_tocache
        self.sensitivity = sensitivity  # Parameter 's' in paper
        
        # SPyNet for optical flow estimation (lightweight)
        self.spynet = SPyNet()
        
        # Query, Key, Value projections for masked attention
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, 
                                     padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
    def compute_flow_mask(self, current_frame, ref_frame, sensitivity=None):
        """
        Compute binary mask using forward-backward consistency check.
        
        Args:
            current_frame: Current frame (B, C, H, W)
            ref_frame: Reference frame (B, C, H, W)
            sensitivity: Sensitivity parameter 's' (default uses self.sensitivity)
            
        Returns:
            mask: Binary mask where 1 indicates unreliable regions (B, 1, H, W)
            flow_forward: Forward optical flow
            flow_backward: Backward optical flow
        """
        if sensitivity is None:
            sensitivity = self.sensitivity
            
        # Compute bidirectional optical flow
        flow_forward = self.spynet(current_frame, ref_frame)  # t-1 -> t
        flow_backward = self.spynet(ref_frame, current_frame)  # t -> t-1
        
        # Forward-backward consistency check (Eq. 3-4 in paper)
        warped_to_ref = warp(current_frame, flow_forward)
        warped_back = warp(warped_to_ref, flow_backward)
        
        # Compute absolute difference
        diff = torch.abs(warped_back - current_frame)
        diff_map = torch.mean(diff, dim=1, keepdim=True)  # Average over channels
        
        # Apply sensitivity threshold (Eq. 5 in paper)
        # M(i,j) = 1 if s * D(i,j) / 255 > 0.5, else 0
        threshold = 0.5 * 255.0 / sensitivity
        mask = (diff_map > threshold).float()
        
        return mask, flow_forward, flow_backward
        
    def forward(self, current_feat, ref_feat, sensitivity=None):
        """
        FGMA forward pass.
        
        Args:
            current_feat: Current frame features (B, C, H, W)
            ref_feat: Reference frame features (B, C, H, W)
            sensitivity: Optional sensitivity override for dynamic adjustment
            
        Returns:
            aligned_feat: Aligned features (B, C+C+1, H, W) 
                         = Concat(warped_feat, mask, attention_feat)
        """
        B, C, H, W = current_feat.shape
        
        # Step 1: Compute mask and optical flow
        mask, flow_fwd, flow_bwd = self.compute_flow_mask(
            current_feat, ref_feat, sensitivity
        )
        
        # Step 2: Warp reference features using optical flow
        warped_feat = warp(ref_feat, flow_bwd)  # Warp ref to current
        
        # Step 3: Compute attention ONLY in masked (unreliable) regions
        # Generate Q, K, V
        qkv_current = self.qkv_dwconv(self.qkv(current_feat))
        q, _, _ = qkv_current.chunk(3, dim=1)
        
        qkv_ref = self.qkv_dwconv(self.qkv(ref_feat))
        _, k, v = qkv_ref.chunk(3, dim=1)
        
        # Apply mask to query (only compute attention in unreliable regions)
        q_masked = q * mask
        
        # Reshape for multi-head attention
        q_masked = rearrange(q_masked, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        # Normalize
        q_masked = torch.nn.functional.normalize(q_masked, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        # Attention computation (Eq. 7 in paper)
        attn = (q_masked @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        
        # Reshape back
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        attention_feat = self.project_out(out)
        
        # Step 4: Concatenate outputs (Eq. 8 in paper)
        # Output: [warped_features, mask, attention_features]
        aligned_feat = torch.cat([warped_feat, mask, attention_feat], dim=1)
        
        return aligned_feat

##########################################################################
## Normal Attentions
class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()
        self.dim = dim
        self.bias = bias
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, k_cached=None, v_cached=None):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out, None, None

class ReducedAttn(nn.Module):
    def __init__(self, c, DW_Expand=2.0, drop_out_rate=0.):
        super().__init__()
        dw_channel = int(c * DW_Expand)
        self.conv1 = nn.Conv2d(in_channels=c, 
                               out_channels=dw_channel, 
                               kernel_size=1, 
                               padding=0, 
                               stride=1, 
                               groups=1, 
                               bias=True)
        
        self.conv2 = nn.Conv2d(in_channels=dw_channel, 
                               out_channels=dw_channel, 
                               kernel_size=3, 
                               padding=1, 
                               stride=1, 
                               groups=dw_channel,
                               bias=True)
        
        self.conv3 = nn.Conv2d(in_channels=dw_channel, 
                               out_channels=c, 
                               kernel_size=1, 
                               padding=0, 
                               stride=1, 
                               groups=1, 
                               bias=True)
        
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, k_cached=None, v_cached=None):
        x = self.conv1(inp)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        return x * self.beta, None, None

##########################################################################
class DeAltHDRBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type, num_heads=1, 
                 attention_type='channel', FFW_type="GFFW", num_frames_tocache=4, sensitivity=15.0):
        super(DeAltHDRBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attention_type = attention_type

        if attention_type == "Channel":
            self.attn = ChannelAttention(dim, num_heads, bias)
        elif attention_type == "ReducedAttn":
            self.attn = ReducedAttn(dim)
        elif attention_type == "FHR":  # Caches num_frames_tocache
            self.attn = FrameHistoryRouter(dim, num_heads, bias, num_frames_tocache)
        elif attention_type == "FGMA":  # Flow-Guided Masked Attention
            self.attn = FlowGuidedMaskedAttention(dim, num_heads, bias, num_frames_tocache, sensitivity)
        elif attention_type == "NoAttn":
            self.attn = None
        else:
            print(attention_type, " Not defined")
            exit()
            
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        if FFW_type == "GFFW":
            self.ffn = GatedFeedForward(dim, ffn_expansion_factor, bias)
        elif FFW_type == "FFW":
            self.ffn = FeedForward(dim)
        else:
            print(FFW_type, " Not defined")
            exit()

    def forward(self, x, k_cached=None, v_cached=None, ref_feat=None, sensitivity=None):
        """
        Args:
            x: Current frame features
            k_cached, v_cached: Cached keys/values for FHR
            ref_feat: Reference frame features (for FGMA)
            sensitivity: Dynamic sensitivity parameter (for FGMA)
        """
        if self.attn is None:
            return x + self.ffn(self.norm2(x)), None, None
        else:
            if self.attention_type == "FGMA":
                # FGMA requires reference features
                if ref_feat is not None:
                    attn_out = self.attn(self.norm1(x), self.norm1(ref_feat), sensitivity)
                    # FGMA returns concatenated features, need to project back
                    # attn_out shape: (B, C+C+1, H, W)
                    # We take only the attention part or process it
                    # For simplicity, we can use a projection layer or just take channels
                    # Here we'll sum the components (simplified)
                    B, C_cat, H, W = attn_out.shape
                    expected_C = x.shape[1]
                    # Split the concatenated output
                    warped_feat = attn_out[:, :expected_C, :, :]
                    mask = attn_out[:, expected_C:expected_C+1, :, :]
                    attention_feat = attn_out[:, expected_C+1:, :, :]
                    # Combine using mask
                    attn_out = mask * attention_feat + (1 - mask) * warped_feat
                else:
                    attn_out = torch.zeros_like(x)
                k_tocache, v_tocache = None, None
            elif self.attention_type == "FHR":
                attn_out, k_tocache, v_tocache = self.attn(self.norm1(x), k_cached, v_cached)
            else:
                attn_out, k_tocache, v_tocache = self.attn(self.norm1(x), k_cached, v_cached)
                
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
            return x, k_tocache, v_tocache

class LevelBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type, num_blocks,
                  attn_type1="Channel", attn_type2="FHR", FFW_type="GFFW", 
                  num_frames_tocache=4, num_heads=1, sensitivity=15.0):
        super(LevelBlock, self).__init__()
        self.num_blocks = num_blocks
        self.attn_type2 = attn_type2
        Block_list = []
            
        for _ in range(num_blocks - 1):
            Block_list.append(DeAltHDRBlock(dim=dim, num_heads=num_heads, 
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias, 
                             LayerNorm_type=LayerNorm_type, attention_type=attn_type1, 
                             FFW_type=FFW_type, num_frames_tocache=num_frames_tocache,
                             sensitivity=sensitivity))
            
        Block_list.append(DeAltHDRBlock(dim=dim, num_heads=num_heads, 
                        ffn_expansion_factor=ffn_expansion_factor, bias=bias, 
                        LayerNorm_type=LayerNorm_type, attention_type=attn_type2, 
                        FFW_type=FFW_type, num_frames_tocache=num_frames_tocache,
                        sensitivity=sensitivity))
            
        self.transformer_blocks = nn.ModuleList(Block_list)

    def forward(self, x, k_cached=None, v_cached=None, ref_feat=None, sensitivity=None):
        """
        Args:
            x: Current frame features
            k_cached, v_cached: Cached keys/values
            ref_feat: Reference frame features (for FGMA in last block)
            sensitivity: Dynamic sensitivity parameter
        """
        for i in range(self.num_blocks - 1):
            x, _, _ = self.transformer_blocks[i](x, k_cached, v_cached)

        # Pass reference features and sensitivity to the last block (if it uses FGMA)
        if self.attn_type2 == "FGMA":
            out1, k_tocache, v_tocache = self.transformer_blocks[-1](
                x, k_cached, v_cached, ref_feat=ref_feat, sensitivity=sensitivity
            )
        else:
            out1, k_tocache, v_tocache = self.transformer_blocks[-1](x, k_cached, v_cached)
            
        if k_tocache != None:
            return out1, k_tocache, v_tocache
        else:
            return out1, None, None

##########################################################################
class DeAltHDR(nn.Module):
    def __init__(self,
        inp_channels,
        out_channels,
        dim,
        Enc_blocks,
        Middle_blocks,
        Dec_blocks,
        num_heads_blks, 
        num_refinement_blocks,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,

        # Encoder attention types
        encoder1_attn_type1, encoder1_attn_type2,
        encoder2_attn_type1, encoder2_attn_type2,
        encoder3_attn_type1, encoder3_attn_type2,

        # Decoder attention types
        decoder1_attn_type1, decoder1_attn_type2,
        decoder2_attn_type1, decoder2_attn_type2,
        decoder3_attn_type1, decoder3_attn_type2,

        # FFW types for each encoder and decoder level
        encoder1_ffw_type, encoder2_ffw_type, encoder3_ffw_type,
        decoder1_ffw_type, decoder2_ffw_type, decoder3_ffw_type,

        # Latent
        latent_attn_type1, latent_attn_type2, latent_attn_type3, latent_ffw_type,

        # Refinement
        refinement_attn_type1, refinement_attn_type2, refinement_ffw_type,

        use_both_input,
        num_frames_tocache,
        # New parameters for dual encoder and training modes
        use_dual_encoder=True,
        training_mode='mixed',  # 'optical_flow', 'attention', 'mixed'
        sensitivity=15.0):      # Default sensitivity parameter
        super(DeAltHDR, self).__init__()
        if use_both_input:
            inp_channels *= 2
        self.use_both_input = use_both_input
        self.use_dual_encoder = use_dual_encoder
        self.training_mode = training_mode
        self.num_heads = num_heads_blks
        self.sensitivity = sensitivity
        
        # Dual encoder for long/short exposure
        if use_dual_encoder:
            # Long exposure encoder
            self.long_exposure_projection = nn.Conv2d(inp_channels, 
                                                   dim, kernel_size=3, 
                                                   stride=1, padding=1, 
                                                   bias=bias)
            # Short exposure encoder  
            self.short_exposure_projection = nn.Conv2d(inp_channels, 
                                                    dim, kernel_size=3, 
                                                    stride=1, padding=1, 
                                                    bias=bias)
        else:
            self.input_projection = nn.Conv2d(inp_channels, 
                                         dim, kernel_size=3, 
                                         stride=1, padding=1, 
                                         bias=bias) 
        
        # Encoder Levels
        self.encoder_level1 = LevelBlock(dim=dim, bias=bias, ffn_expansion_factor=ffn_expansion_factor,
                                          LayerNorm_type=LayerNorm_type, num_blocks=Enc_blocks[0],
                                          attn_type1=encoder1_attn_type1, attn_type2=encoder1_attn_type2,
                                          FFW_type=encoder1_ffw_type, num_frames_tocache=num_frames_tocache, 
                                          num_heads=self.num_heads[0], sensitivity=sensitivity)
        
        self.down1_2 = Downsample(dim)  # From Level 1 to Level 2
        self.encoder_level2 = LevelBlock(dim=int(dim*2**1), bias=bias, ffn_expansion_factor=ffn_expansion_factor,
                                          LayerNorm_type=LayerNorm_type, num_blocks=Enc_blocks[1],
                                          attn_type1=encoder2_attn_type1, attn_type2=encoder2_attn_type2,
                                          FFW_type=encoder2_ffw_type, num_frames_tocache=num_frames_tocache, 
                                          num_heads=self.num_heads[1], sensitivity=sensitivity)
        
        self.down2_3 = Downsample(int(dim*2**1))  # From Level 2 to Level 3
        self.encoder_level3 = LevelBlock(dim=int(dim*2**2), bias=bias, ffn_expansion_factor=ffn_expansion_factor,
                                          LayerNorm_type=LayerNorm_type, num_blocks=Enc_blocks[2],
                                          attn_type1=encoder3_attn_type1, attn_type2=encoder3_attn_type2, 
                                          FFW_type=encoder3_ffw_type, num_frames_tocache=num_frames_tocache, 
                                          num_heads=self.num_heads[2], sensitivity=sensitivity)

        # Middle block
        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = LevelBlock(dim=int(dim*2**3), ffn_expansion_factor=ffn_expansion_factor,
                                       bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=Middle_blocks,
                                       attn_type1=latent_attn_type1, attn_type2=latent_attn_type2, 
                                       FFW_type=latent_ffw_type, 
                                       num_frames_tocache=num_frames_tocache, num_heads=self.num_heads[3],
                                       sensitivity=sensitivity)

        # Decoder Levels
        self.up4_3 = Upsample(int(dim*2**3))  # From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = LevelBlock(dim=int(dim*2**2), ffn_expansion_factor=ffn_expansion_factor,
                                         bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=Dec_blocks[0],
                                         attn_type1=decoder1_attn_type1, attn_type2=decoder1_attn_type2, 
                                         FFW_type=decoder1_ffw_type, num_frames_tocache=num_frames_tocache, 
                                         num_heads=self.num_heads[2], sensitivity=sensitivity)
        
        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = LevelBlock(dim=int(dim*2**1), ffn_expansion_factor=ffn_expansion_factor,
                                           bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=Dec_blocks[1],
                                           attn_type1=decoder2_attn_type1, attn_type2=decoder2_attn_type2, 
                                           FFW_type=decoder2_ffw_type, num_frames_tocache=num_frames_tocache, 
                                           num_heads=self.num_heads[1], sensitivity=sensitivity)
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1 
        self.reduce_chan_level1 = nn.Conv2d(int(dim*2**1), int(dim*1), kernel_size=1, bias=bias)
        self.decoder_level1 = LevelBlock(dim=int(dim*1), ffn_expansion_factor=ffn_expansion_factor,
                                           bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=Dec_blocks[2],
                                           attn_type1=decoder3_attn_type1, attn_type2=decoder3_attn_type2, 
                                           FFW_type=decoder3_ffw_type, num_frames_tocache=num_frames_tocache, 
                                           num_heads=self.num_heads[0], sensitivity=sensitivity)
        
        # Refinement Block
        self.refinement = LevelBlock(dim=int(dim*1), ffn_expansion_factor=ffn_expansion_factor,
                                     bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_refinement_blocks,
                                     attn_type1=refinement_attn_type1, attn_type2=refinement_attn_type2, 
                                     FFW_type=refinement_ffw_type, num_frames_tocache=num_frames_tocache, 
                                     num_heads=self.num_heads[0], sensitivity=sensitivity)
        
        self.ending = nn.Conv2d(in_channels=int(dim*1),
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 padding=1,
                                 stride=1,
                                 groups=1,
                                 bias=True)
        
        self.padder_size = (2**3)*4

    def forward(self, inp_img_, k_cached=None, v_cached=None, exposure_type=None, training_mode=None, sensitivity=None):
        """
        Forward pass with support for dual encoder and mixed training modes
        
        Args:
            inp_img_: Input frames [B, T, C, H, W] where T=5 for T-2,T-1,T,T+1,T+2
            k_cached: Cached keys (list of caches for each level)
            v_cached: Cached values (list of caches for each level)
            exposure_type: 'long' or 'short' exposure type
            training_mode: 'optical_flow', 'attention', or 'mixed'
            sensitivity: Sensitivity parameter 's' for dynamic FGMA (default: 15.0)
        """
        B, T, C, H, W = inp_img_.shape
        assert T == 5, f"Expected 5 frames (T-2,T-1,T,T+1,T+2), got {T}"
        
        inp_img_ = self.check_image_size(inp_img_)
        
        if k_cached == None:
            k_cached = [None] * 8  # 3 encoders + 1 middle + 3 decoders + 1 refinement
            v_cached = [None] * 8

        # Extract frames: T-2, T-1, T (current), T+1, T+2
        frame_t_minus_2 = inp_img_[:, 0, :, :, :]
        frame_t_minus_1 = inp_img_[:, 1, :, :, :]
        frame_t = inp_img_[:, 2, :, :, :]  # Current frame
        frame_t_plus_1 = inp_img_[:, 3, :, :, :]
        frame_t_plus_2 = inp_img_[:, 4, :, :, :]
        
        # Determine training mode
        if training_mode is None:
            training_mode = self.training_mode
            
        # Select encoder based on exposure type
        if self.use_dual_encoder and exposure_type is not None:
            if exposure_type == 'long':
                inp_enc_level1 = self.long_exposure_projection(frame_t.float())
            else:  # short exposure
                inp_enc_level1 = self.short_exposure_projection(frame_t.float())
        else:
            inp_enc_level1 = self.input_projection(frame_t.float())
        
        # ============ ENCODER ============
        # Level 1
        out_enc_level1, _, _ = self.encoder_level1(inp_enc_level1, k_cached[0], v_cached[0])
        
        # Level 2
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2, _, _ = self.encoder_level2(inp_enc_level2, k_cached[1], v_cached[1])
        
        # Level 3
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3, _, _ = self.encoder_level3(inp_enc_level3, k_cached[2], v_cached[2])

        # ============ MIDDLE (LATENT) ============
        inp_latent = self.down3_4(out_enc_level3)
        latent, k_cached[3], v_cached[3] = self.latent(inp_latent, k_cached[3], v_cached[3])

        # ============ DECODER with FGMA ============
        # For decoder, we need to align neighboring frames
        # Process neighboring frames through encoder to get features
        neighbor_frames = [frame_t_minus_2, frame_t_minus_1, frame_t_plus_1, frame_t_plus_2]
        
        # Decoder Level 3
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        
        # Align neighboring frames at level 3 if using FGMA
        if hasattr(self.decoder_level3, 'attn_type2') and self.decoder_level3.attn_type2 == "FGMA":
            # Process one neighbor for alignment (e.g., T-1)
            if self.use_dual_encoder and exposure_type is not None:
                # Use opposite encoder for neighbor (alternating exposure)
                neighbor_exposure = 'short' if exposure_type == 'long' else 'long'
                if neighbor_exposure == 'long':
                    neighbor_feat_l1 = self.long_exposure_projection(frame_t_minus_1.float())
                else:
                    neighbor_feat_l1 = self.short_exposure_projection(frame_t_minus_1.float())
            else:
                neighbor_feat_l1 = self.input_projection(frame_t_minus_1.float())
            
            # Encode neighbor to level 3
            neighbor_enc_l1, _, _ = self.encoder_level1(neighbor_feat_l1, None, None)
            neighbor_enc_l2 = self.down1_2(neighbor_enc_l1)
            neighbor_enc_l2, _, _ = self.encoder_level2(neighbor_enc_l2, None, None)
            neighbor_enc_l3 = self.down2_3(neighbor_enc_l2)
            neighbor_enc_l3, _, _ = self.encoder_level3(neighbor_enc_l3, None, None)
            
            # Use FGMA to align
            out_dec_level3, k_cached[4], v_cached[4] = self.decoder_level3(
                inp_dec_level3, k_cached[4], v_cached[4], 
                ref_feat=neighbor_enc_l3, sensitivity=sensitivity
            )
        else:
            out_dec_level3, k_cached[4], v_cached[4] = self.decoder_level3(
                inp_dec_level3, k_cached[4], v_cached[4]
            )
        
        # Decoder Level 2
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2, k_cached[5], v_cached[5] = self.decoder_level2(
            inp_dec_level2, k_cached[5], v_cached[5]
        )
        
        # Decoder Level 1
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1, k_cached[6], v_cached[6] = self.decoder_level1(
            inp_dec_level1, k_cached[6], v_cached[6]
        )
        
        # ============ REFINEMENT ============
        out_refinement, k_cached[7], v_cached[7] = self.refinement(
            out_dec_level1, k_cached[7], v_cached[7]
        )
        
        # ============ OUTPUT ============
        out = self.ending(out_refinement)
        
        # Crop to original size
        _, _, _, h_orig, w_orig = inp_img_.size()
        out = out[:, :, :h_orig, :w_orig]
        
        return out, k_cached, v_cached
    
    def check_image_size(self, x):
        _, _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    from basicsr.utils.options import parse

    opt = parse(
                "options/DeAltHDR.yml",
                 is_train=True)

    model = create_video_model(opt)

    inp_shape = (5, 3, 256, 256)  # T-2,T-1,T,T+1,T+2
    macs, params = get_model_complexity_info(model, inp_shape, 
                                             verbose=False, 
                                             print_per_layer_stat=False)
    print(f"MACs: {macs}")
    print(f"Params: {params}")


