"""
Spectra2TS: Vision Transformer (ViT) to Time Series forecasting model
Integrates STFT spectrogram generation, rectangular ViT encoder, and Transformer decoder with cross-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import numpy as np
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from layers.Embed import DataEmbedding
from layers.Transformer_EncDec import Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer

def get_STFT_spectra(tensor_data, target_width=None, device='cuda') -> torch.Tensor:
    """
    Generate STFT spectrograms with configurable width using PyTorch.
    
    Args:
        tensor_data: Input time series data tensor of shape (time_steps, features) or numpy array
        target_width: Desired width of spectrogram (should match context_length)
                     If None, uses automatic calculation
        device: Device to run computations on ('cuda' or 'cpu')
    
    Returns:
        Spectrogram tensor of shape (num_features, 128, 128)
    """
    
    # Move to specified device
    tensor_data = tensor_data.to(device)
    
    # Transpose to get shape (features, time_steps)
    tensor_data = tensor_data.T
    
    # Get time series length
    time_length = tensor_data.shape[1]
    
    # If target_width is specified, use it; otherwise use time_length
    if target_width is None:
        target_width = time_length
    
    # Calculate STFT parameters
    n_fft = 64  # This gives us 33 frequency bins (we'll use 32)
    
    # Calculate nperseg and noverlap to achieve target_width time frames (same logic as original)
    if target_width >= time_length:
        # If target width is larger than signal, use small window
        nperseg = min(16, time_length)
        noverlap = 0
    else:
        # Calculate appropriate window size
        nperseg = min(time_length // 4, 64)  # Don't make window too large
        # Calculate noverlap to get approximately target_width frames
        # Rearranging the formula: noverlap = nperseg - (signal_length - nperseg) / (target_width - 1)
        if target_width > 1:
            noverlap = int(nperseg - (time_length - nperseg) / (target_width - 1))
            noverlap = max(0, min(noverlap, nperseg - 1))
        else:
            noverlap = 0
    
    # Convert to PyTorch parameters
    win_length = nperseg
    hop_length = nperseg - noverlap
    
    # Create window tensor on the same device
    window = torch.hann_window(win_length, device=device)
    
    # Initialize list to store spectra
    spectra_list = []
    
    # Process each feature (row in tensor_data)
    for i in range(tensor_data.shape[0]):
        signal_tensor = tensor_data[i]
        
        # Apply STFT
        stft_result = torch.stft(
            signal_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            pad_mode='constant',
            return_complex=True
        )
        
        # Get magnitude spectrum
        spec = torch.abs(stft_result)
        
        # Take only first 32 frequency bins
        spec = spec[:32, :]
        
        # Resize time dimension to exactly target_width using interpolation
        if spec.shape[1] != target_width:
            # Use PyTorch's interpolate function
            spec_expanded = spec.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 32, time)
            
            # Interpolate to target width
            spec_resized = F.interpolate(
                spec_expanded,
                size=(32, target_width),
                mode='bilinear',
                align_corners=False
            )
            
            spec = spec_resized[0, 0]  # Direct indexing instead of squeeze
        
        # Ensure the shape is exactly (32, target_width)
        if spec.shape[0] > 32:
            spec = spec[:32, :]
        elif spec.shape[0] < 32:
            # Pad with zeros if needed
            padding_size = 32 - spec.shape[0]
            padding = torch.zeros(padding_size, spec.shape[1], device=device)
            spec = torch.cat([spec, padding], dim=0)
        
        # Resize spectrum to 128x128 using bilinear interpolation
        spec_expanded = spec.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 32, target_width)
        spec_final = F.interpolate(
            spec_expanded,
            size=(128, 128),
            mode='bilinear',
            align_corners=False
        )
        spec_final = spec_final[0, 0]  # Direct indexing instead of squeeze
        
        spectra_list.append(spec_final)
    
    # Stack all spectra along first dimension
    spectra = torch.stack(spectra_list, dim=0)
    
    return spectra

# ============================================================================
# ViT Encoder Components (from vit_encoder.py)
# ============================================================================

class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them."""
    
    def __init__(self, image_height=64, image_width=64, patch_size=8, in_channels=3, embed_dim=512):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.num_patches_h = image_height // patch_size
        self.num_patches_w = image_width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )
        
    def forward(self, x):
        return self.projection(x)


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for rectangular images."""
    
    def __init__(self, embed_dim, max_height=100, max_width=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create 2D positional encoding
        pe = torch.zeros(max_height, max_width, embed_dim)
        
        # Height position encoding (uses half of embed_dim)
        pos_h = torch.arange(0, max_height).unsqueeze(1).float()
        pos_w = torch.arange(0, max_width).unsqueeze(1).float()
        
        # Divide embed_dim into two parts for height and width
        d_h = embed_dim // 2
        d_w = embed_dim - d_h
        
        div_term_h = torch.exp(torch.arange(0, d_h, 2).float() * 
                               -(math.log(10000.0) / d_h))
        div_term_w = torch.exp(torch.arange(0, d_w, 2).float() * 
                               -(math.log(10000.0) / d_w))
        
        # Height encoding
        pe[:, :, 0:d_h:2] = torch.sin(pos_h * div_term_h).unsqueeze(1).expand(-1, max_width, -1)
        pe[:, :, 1:d_h:2] = torch.cos(pos_h * div_term_h).unsqueeze(1).expand(-1, max_width, -1)
        
        # Width encoding
        pe[:, :, d_h::2] = torch.sin(pos_w * div_term_w).unsqueeze(0).expand(max_height, -1, -1)
        pe[:, :, d_h+1::2] = torch.cos(pos_w * div_term_w).unsqueeze(0).expand(max_height, -1, -1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, num_patches_h, num_patches_w, device=None):
        """Get positional encoding for specific patch grid size."""
        pos_enc = self.pe[:num_patches_h, :num_patches_w, :]
        if device is not None:
            pos_enc = pos_enc.to(device)
        pos_enc = rearrange(pos_enc, 'h w d -> (h w) d')
        return pos_enc


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, (f"embed_dim = {embed_dim} should be divisible by num_heads = {num_heads}")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Aggregate
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x


class FeedForward(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm."""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout)
        self.resweight = nn.Parameter(torch.Tensor([0]))
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x)) * self.resweight
        x = x + self.linear1(self.ffn(self.norm2(x))) * self.resweight
        return x


class RectangularViT(nn.Module):
    """Vision Transformer with support for rectangular images."""
    
    def __init__(
        self,
        image_height=128,
        image_width=128,
        patch_size=8,
        in_channels=1,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        embed_dropout=0.1,
    ):
        super().__init__()
        
        # Validate image dimensions
        assert image_height % patch_size == 0, f"Image height {image_height} must be divisible by patch size {patch_size}"
        assert image_width % patch_size == 0, f"Image width {image_width} must be divisible by patch size {patch_size}"
        
        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.num_patches_h = image_height // patch_size
        self.num_patches_w = image_width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_height, image_width, patch_size, in_channels, embed_dim
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional encoding (2D for rectangular support)
        self.pos_encoding = PositionalEncoding2D(
            embed_dim, 
            max_height=100,  # Support up to 100x100 patches
            max_width=100,
            dropout=embed_dropout
        )
        
        self.embed_dropout = nn.Dropout(embed_dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.cls_token, std=0.04)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.04)
                if module.bias is not None:
                    nn.init.trunc_normal_(module.bias, std=0.04)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.trunc_normal_(module.bias, std=0.04)
                
    def get_last_hidden_state(self, x):
        """Get the last hidden state (all tokens) from the transformer."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        pos_enc = self.pos_encoding(self.num_patches_h, self.num_patches_w, device=x.device)
        pos_enc = torch.cat([torch.zeros_like(cls_tokens[0, 0, :]).unsqueeze(0), pos_enc], dim=0)
        x = x + pos_enc.unsqueeze(0).to(x.device)
        
        x = self.embed_dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        return x  # Return all tokens including CLS

# ============================================================================
# Main Spectra2TS Model (TSLib Compatible)
# ============================================================================

class Model(nn.Module):
    """
    Spectra2TS: Vision Transformer (ViT) to Time Series forecasting model
    TSLib compatible interface for long-term forecasting and classification tasks
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # Store task and basic config
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len  # TSLib context length
        self.pred_len = configs.pred_len  # TSLib prediction length
        self.label_len = configs.label_len  # TSLib label length (for decoder input)
        
        # Map TSLib parameters to model parameters with defaults
        self.prediction_length = configs.pred_len
        self.context_length = configs.seq_len
        self.num_channels = configs.enc_in  # Use enc_in as number of input features
        
        # Default values from spectra2ts (use if not in configs)
        self.d_model = getattr(configs, 'd_model', 512)
        self.ts_num_heads = getattr(configs, 'n_heads', 8)
        self.ts_num_layers = getattr(configs, 'd_layers', 3)
        self.ts_dim_feedforward = getattr(configs, 'd_ff', 1024)
        self.ts_dropout = getattr(configs, 'dropout', 0.1)
        self.c_out = getattr(configs, 'c_out', 1)  # Output dimension (1 for univariate)
        
        # Rectangular ViT Encoder (128x128 spectrograms)
        self.vit_encoder = RectangularViT(
            image_height=128,  # Fixed for resized spectrograms
            image_width=128,   # Fixed for resized spectrograms  
            in_channels=self.num_channels,
            embed_dim=512,
            depth=3,
            num_heads=8,
            mlp_ratio=4,
            dropout=0.1
        )
        #self.dsw_embedding = DSW_embedding2(seg_len=self.seq_len, d_model=self.d_model)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
        
        self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Sequential(
                    nn.Linear(configs.d_model, configs.d_model // 2),
                    nn.GELU(),
                    nn.Dropout(configs.dropout),
                    nn.Linear(configs.d_model // 2, configs.c_out)
                )
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Long-term forecasting using ViT encoder and Transformer decoder."""
        device = next(self.parameters()).device
        batch_size = x_enc.size(0)
        
        # Series Stationarization adopted from NSformer
        mean_enc = x_enc.mean(1, keepdim=True).detach() # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc


        spectra_list = []
        for item in x_enc:
            spectra = get_STFT_spectra(item, device=device)
            spectra_list.append(spectra)
        spectra_tensor = torch.stack(spectra_list, dim=0)  # (batch, channels, 128, 128)
        
        # memory: using encoded spectrogram features
        enc_out = self.vit_encoder.get_last_hidden_state(spectra_tensor)  # (batch, num_patches+1, 512)

        # generate starting tokens for decoder
        context_values = x_enc[:, -self.seq_len:, :]
        starts = self.dec_embedding(context_values, None)  # (batch, label_len, 512) -> (batch, c_out, 512)

        dec_out = torch.cat([starts, torch.zeros(batch_size, self.pred_len, self.d_model).to(device)], dim=1)

        predictions = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        
        return predictions

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        TSLib compatible forward method.
        
        Args:
            x_enc: Encoder input [B, seq_len, features]
            x_mark_enc: Encoder time features [B, seq_len, time_features] (ignored)
            x_dec: Decoder input [B, label_len+pred_len, features] (used for teacher forcing target)
            x_mark_dec: Decoder time features [B, label_len+pred_len, time_features] (ignored)
            mask: Mask for imputation (not used)
            
        Returns:
            Output predictions or classifications
        """
        
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, pred_len, 1] - univariate output (last feature)
        else:
            raise ValueError(f"Task {self.task_name} not supported. Only 'long_term_forecast' and 'classification' are supported.")
