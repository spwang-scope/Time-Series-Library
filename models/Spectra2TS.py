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


# ============================================================================
# STFT Spectrogram Generation (from pytorch_stft.py)
# ============================================================================

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
    
    # Convert to tensor if numpy array
    if isinstance(tensor_data, np.ndarray):
        tensor_data = torch.from_numpy(tensor_data).float()
    
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
    
    def __init__(self, image_height=64, image_width=96, patch_size=8, in_channels=3, embed_dim=768):
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
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RectangularViT(nn.Module):
    """Vision Transformer with support for rectangular images."""
    
    def __init__(
        self,
        image_height=128,
        image_width=128,
        patch_size=8,
        in_channels=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
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


def create_rectangular_vit(image_height=128, image_width=128, **kwargs):
    """Factory function to create a rectangular ViT model."""
    # Optimize patch size based on image width
    if image_width in [96, 192]:
        patch_size = 8
    elif image_width in [336]:
        patch_size = 8  # Could also use 12 for fewer patches
    elif image_width in [720]:
        patch_size = 16  # Larger patches for very wide images
    else:
        patch_size = 8  # Default
        
    return RectangularViT(
        image_height=image_height,
        image_width=image_width,
        patch_size=patch_size,
        **kwargs
    )


# ============================================================================
# Transformer Decoder Components (from model.py)
# ============================================================================

class DecoderPositionalEncoding(nn.Module):
    """Dynamic positional encoding for transformer decoder that computes encodings on-the-fly."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        
        # Pre-compute div_term for efficiency (this doesn't depend on sequence length)
        self.register_buffer('div_term', torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        ))
        
        # Cache for computed positional encodings to avoid recomputation
        self._pe_cache = {}
    
    def _compute_pe(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute positional encoding for given sequence length."""
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")
            
        # Check cache first
        cache_key = (seq_len, device.type, device.index if device.index is not None else 0)
        if cache_key in self._pe_cache:
            return self._pe_cache[cache_key]
        
        # Compute positional encoding
        pe = torch.zeros(seq_len, self.d_model, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = self.div_term.to(device)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)
        
        # Cache the result (limit cache size to prevent memory issues)
        if len(self._pe_cache) < 100:  # Reasonable cache limit
            self._pe_cache[cache_key] = pe
            
        return pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: Tensor of shape (batch_size, seq_length, d_model)"""
        seq_len = x.size(1)
        device = x.device
        
        # Get or compute positional encoding for this sequence length
        pe = self._compute_pe(seq_len, device)
        
        # Add positional encoding and apply dropout
        x = x + pe
        return self.dropout(x)


class KVCacheCrossAttention(nn.Module):
    """Cross-attention with KV-caching for efficient autoregressive decoding."""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model {d_model} must be divisible by nhead {nhead}"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = self.d_head ** -0.5
        
        # Separate linear projections for Q, K, V (like CATS)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,           # Query: [batch, seq_len, d_model] (seq_len=1 for inference, >1 for teacher forcing)
        encoder_kv: torch.Tensor,      # Static encoder K,V [batch, 258, d_model]
        decoder_kv_cache: Optional[torch.Tensor] = None,  # Cached decoder K,V [batch, steps, d_model] (None for teacher forcing)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: Query tensor [batch, seq_len, d_model] (seq_len=1 for inference, pred_len for teacher forcing)
            encoder_kv: Static encoder memory [batch, 258, d_model] 
            decoder_kv_cache: Accumulated decoder K,V [batch, current_steps, d_model] (None for teacher forcing)
            
        Returns:
            output: Attention output [batch, seq_len, d_model]
            new_kv: New K,V to add to cache [batch, 1, d_model] for inference, None for teacher forcing
        """
        batch_size, seq_len, _ = query.shape
        
        # Project queries
        q = self.W_Q(query)  # [batch, seq_len, d_model]
        q = q.view(batch_size, seq_len, self.nhead, self.d_head)  # [batch, seq_len, nhead, d_head]
        q = q.transpose(1, 2)  # [batch, nhead, seq_len, d_head]
        
        # Project encoder K,V (static, computed once)
        encoder_k = self.W_K(encoder_kv)  # [batch, 258, d_model]
        encoder_v = self.W_V(encoder_kv)  # [batch, 258, d_model]
        
        encoder_k = encoder_k.view(batch_size, -1, self.nhead, self.d_head).transpose(1, 2)  # [batch, nhead, 258, d_head]
        encoder_v = encoder_v.view(batch_size, -1, self.nhead, self.d_head).transpose(1, 2)  # [batch, nhead, 258, d_head]
        
        if decoder_kv_cache is None:
            # Teacher forcing mode: use encoder K,V + query K,V
            query_k = self.W_K(query)  # [batch, seq_len, d_model]
            query_v = self.W_V(query)  # [batch, seq_len, d_model]
            
            query_k = query_k.view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2)  # [batch, nhead, seq_len, d_head]
            query_v = query_v.view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2)  # [batch, nhead, seq_len, d_head]
            
            # Concatenate: encoder + current sequence
            k = torch.cat([encoder_k, query_k], dim=2)  # [batch, nhead, 258+seq_len, d_head]
            v = torch.cat([encoder_v, query_v], dim=2)  # [batch, nhead, 258+seq_len, d_head]
            
            new_kv = None  # No caching in teacher forcing mode
        else:
            # Inference mode: KV-caching (seq_len should be 1)
            assert seq_len == 1, f"In inference mode, seq_len should be 1, got {seq_len}"
            
            # Project current query to get new K,V for decoder cache
            new_k = self.W_K(query)  # [batch, 1, d_model]
            new_v = self.W_V(query)  # [batch, 1, d_model]
            
            new_k_reshaped = new_k.view(batch_size, 1, self.nhead, self.d_head).transpose(1, 2)  # [batch, nhead, 1, d_head]
            new_v_reshaped = new_v.view(batch_size, 1, self.nhead, self.d_head).transpose(1, 2)  # [batch, nhead, 1, d_head]
            
            # Project cached decoder K,V
            cached_k = self.W_K(decoder_kv_cache)  # [batch, steps, d_model]
            cached_v = self.W_V(decoder_kv_cache)  # [batch, steps, d_model]
            
            cached_k = cached_k.view(batch_size, -1, self.nhead, self.d_head).transpose(1, 2)  # [batch, nhead, steps, d_head]
            cached_v = cached_v.view(batch_size, -1, self.nhead, self.d_head).transpose(1, 2)  # [batch, nhead, steps, d_head]
            
            # Concatenate: encoder + cached_decoder + new
            k = torch.cat([encoder_k, cached_k, new_k_reshaped], dim=2)  # [batch, nhead, 258+steps+1, d_head]
            v = torch.cat([encoder_v, cached_v, new_v_reshaped], dim=2)  # [batch, nhead, 258+steps+1, d_head]
            
            new_kv = query  # Cache the original query for next step
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, nhead, seq_len, total_len]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, nhead, seq_len, d_head]
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, seq_len, nhead, d_head]
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)  # [batch, seq_len, d_model]
        output = self.out_proj(attn_output)
        
        return output, new_kv


class TransformerDecoderLayer(nn.Module):
    """Custom transformer decoder layer with self-attention and cross-attention."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention with KV-caching
        self.cross_attn = KVCacheCrossAttention(d_model, nhead, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(
        self,
        tgt: torch.Tensor,                    # Query [batch, seq_len, d_model] (seq_len=1 for inference, >1 for teacher forcing)
        encoder_memory: torch.Tensor,         # Static encoder K,V [batch, 258, d_model]  
        decoder_kv_cache: Optional[torch.Tensor] = None,  # Cached decoder K,V (None for teacher forcing)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            output: Layer output [batch, seq_len, d_model]
            new_kv: New K,V to add to cache [batch, 1, d_model] for inference, None for teacher forcing
        """
        # Self-attention with residual connection
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=None)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # KV-cached cross-attention
        tgt2, new_kv = self.cross_attn(tgt, encoder_memory, decoder_kv_cache)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward with residual connection
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt, new_kv


class TransformerDecoderWithCrossAttention(nn.Module):
    """Transformer decoder with proper cross-attention mechanism."""
    
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        prediction_length: int = 96,
        context_length: int = 96,
        time_series_dim: int = 1,
        encoder_dim: int = 768,  # ViT encoder output dimension
    ):
        super().__init__()
        
        self.d_model = d_model
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.time_series_dim = time_series_dim
        
        # Embedding for time series values
        self.value_embedding = nn.Linear(time_series_dim, d_model)
        
        # Dynamic positional encoding for variable prediction lengths
        self.pos_encoding = DecoderPositionalEncoding(d_model, dropout=dropout)
        
        # Project encoder output to decoder dimension for cross-attention
        self.encoder_projection = nn.Linear(encoder_dim, d_model)
        
        
        # Custom transformer decoder layers with cross-attention
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, time_series_dim)
        )
        
        # Project context condition to time series dimension for start token
        self.context_to_start_token = nn.Linear(encoder_dim, time_series_dim)
        
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize decoder parameters with better schemes for gradient sensitivity."""
        
        # Initialize value embedding layer
        nn.init.xavier_uniform_(self.value_embedding.weight)
        nn.init.constant_(self.value_embedding.bias, 0.0)  # Zero bias to prevent offset in embeddings
        
        # Initialize encoder projection
        nn.init.xavier_uniform_(self.encoder_projection.weight)
        nn.init.constant_(self.encoder_projection.bias, 0.0)  # Zero bias to prevent offset in projections
        
        # Initialize context to start token projection
        nn.init.xavier_uniform_(self.context_to_start_token.weight)
        nn.init.constant_(self.context_to_start_token.bias, 0.0)  # Zero bias for start token generation
        
        # Initialize output projection with smaller weights for stable training
        for i, layer in enumerate(self.output_projection):
            if isinstance(layer, nn.Linear):
                if i == len(self.output_projection) - 1:  # Final layer
                    # Smaller initialization for final output layer - zero bias to avoid systematic offset
                    nn.init.xavier_uniform_(layer.weight, gain=0.01)
                    nn.init.constant_(layer.bias, 0.0)  # Zero bias to prevent systematic offset
                else:
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)  # Zero bias for intermediate layers
        
    
    def forward(
        self, 
        encoder_output: torch.Tensor,
        context_condition: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        use_teacher_forcing: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with optional teacher forcing.
        
        Args:
            encoder_output: Output from ViT encoder (batch_size, num_patches+1, encoder_dim)
            context_condition: Last feature column of context (batch_size, context_length)
            target: Ground truth for teacher forcing (batch_size, prediction_length, time_series_dim)
            use_teacher_forcing: Whether to use teacher forcing (True during training)
            
        Returns:
            Predictions (batch_size, prediction_length, time_series_dim)
        """
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Project encoder output for cross-attention K, V (768 -> 128)
        memory = self.encoder_projection(encoder_output)  # (batch_size, num_patches+1, d_model=128)
        
        # Project context condition to match d_model dimension for concatenation (128 -> 128)
        context_condition_projected = self.encoder_projection(context_condition)  # (batch_size, 1, d_model=128)
        
        # Combine encoder output with context condition for cross-attention
        encoder_memory = torch.cat([memory, context_condition_projected], dim=1)  # (batch_size, num_patches+2, d_model=128)
        
        if use_teacher_forcing and target is not None:
            # Teacher forcing: use ground truth as input, properly aligned for prediction
            # Input: [start_token, target[0], target[1], ..., target[n-2]]
            # Output: [target[0], target[1], target[2], ..., target[n-1]]

            # Generate conditional start token from context condition (CLS token)
            start_tokens = self.context_to_start_token(context_condition)  # [batch, 1, time_series_dim]
            
            # DEBUG: Compare start token with first target value
            first_target = target[:, 0, :].mean()
            start_token_mean = start_tokens.mean()
            print(f"[DEBUG TF START] Start token mean: {start_token_mean:.4f}, First target mean: {first_target:.4f}, Diff: {(start_token_mean - first_target):.4f}")
            
            # Use target[:-1] (all but last element) to predict target (all elements)
            decoder_input = torch.cat([start_tokens, target[:, :-1, :]], dim=1)  # (batch_size, pred_len, ts_dim)
            
            # Embed and add positional encoding
            decoder_input = self.value_embedding(decoder_input)  # (batch_size, pred_len, d_model)
            decoder_input = self.pos_encoding(decoder_input)
            
            # Pass through decoder layers (teacher forcing mode - process entire sequence)
            output = decoder_input
            for layer in self.decoder_layers:
                # For teacher forcing, we don't use KV-caching, just pass None for decoder_kv_cache
                output, _ = layer(output, encoder_memory, decoder_kv_cache=None)
            
            # Project to output dimension - now directly predicts target
            output = self.output_projection(output)  # (batch_size, pred_len, ts_dim)
            
        else:
            # Inference mode: autoregressive generation with KV-caching
            predictions = []
            decoder_kv_cache = None  # Initialize empty cache
            
            # Start with conditional start token from context condition (CLS token)
            current_input = self.context_to_start_token(context_condition)  # [batch, 1, time_series_dim]
            
            # DEBUG: Log start token quality for first step analysis
            print(f"[DEBUG FIRST STEP] Start token range: [{current_input.min():.4f}, {current_input.max():.4f}], mean: {current_input.mean():.4f}")
            
            for step in range(self.prediction_length):
                # Embed current step only (not entire sequence)
                embedded = self.value_embedding(current_input)  # [batch, 1, d_model]
                embedded = self.pos_encoding(embedded)
                
                # Pass through decoder layers with KV-caching (no causal mask needed)
                step_output = embedded
                new_kvs = []
                
                for layer in self.decoder_layers:
                    step_output, new_kv = layer(step_output, encoder_memory, decoder_kv_cache)
                    new_kvs.append(new_kv)
                
                # Update KV cache with new values (use the step_output for next iteration)
                if decoder_kv_cache is None:
                    # First step: initialize cache
                    decoder_kv_cache = step_output  # [batch, 1, d_model]
                else:
                    # Subsequent steps: append to cache
                    decoder_kv_cache = torch.cat([decoder_kv_cache, step_output], dim=1)  # [batch, step+1, d_model]
                
                # Get prediction for next time step
                next_pred = self.output_projection(step_output)  # [batch, 1, time_series_dim]
                predictions.append(next_pred)
                
                # DEBUG: Log first few predictions to analyze sharp drop
                if step < 5:  # First 5 steps
                    print(f"[DEBUG FIRST STEP] Step {step}: input=[{current_input.mean():.4f}], pred=[{next_pred.mean():.4f}], diff=[{(next_pred.mean() - current_input.mean()):.4f}]")
                
                # Update current input for next step
                current_input = next_pred
            
            # Concatenate predictions
            output = torch.cat(predictions, dim=1)  # (batch_size, pred_len, ts_dim)
        
        return output


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
        
        # Map TSLib parameters to model parameters with defaults
        self.prediction_length = configs.pred_len
        self.context_length = configs.seq_len
        self.num_channels = configs.enc_in  # Use enc_in as number of input features
        self.time_series_dim = 1  # Always univariate output (last feature)
        
        # Default values from spectra2ts (use if not in configs)
        self.feature_projection_dim = getattr(configs, 'feature_projection_dim', 128)
        self.ts_model_dim = getattr(configs, 'd_model', 768)
        self.ts_num_heads = getattr(configs, 'n_heads', 8)
        self.ts_num_layers = getattr(configs, 'd_layers', 3)
        self.ts_dim_feedforward = getattr(configs, 'd_ff', 1024)
        self.ts_dropout = getattr(configs, 'dropout', 0.1)
        
        # Teacher forcing mode (controlled by TSLib)
        self.use_teacher_forcing = False  # Default to inference mode
        
        # Rectangular ViT Encoder (128x128 spectrograms)
        self.vit_encoder = create_rectangular_vit(
            image_height=128,  # Fixed for resized spectrograms
            image_width=128,   # Fixed for resized spectrograms  
            in_channels=self.num_channels,
            embed_dim=768,
            depth=2,
            num_heads=12,
            mlp_ratio=4,
            dropout=0.1
        )
        
        # Linear projection for encoder features
        vit_hidden_size = 768  # Default ViT hidden size
        self.encoder_projection = nn.Linear(vit_hidden_size, self.feature_projection_dim)
        
        # Transformer Decoder with Cross-Attention
        self.ts_decoder = TransformerDecoderWithCrossAttention(
            d_model=self.ts_model_dim,
            nhead=self.ts_num_heads,
            num_layers=self.ts_num_layers,
            dim_feedforward=self.ts_dim_feedforward,
            dropout=self.ts_dropout,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            time_series_dim=self.time_series_dim,
            encoder_dim=self.feature_projection_dim,  # After linear projection
        )

    def set_teacher_forcing_mode(self, use_teacher_forcing: bool):
        """Set teacher forcing mode (called by TSLib framework)"""
        self.use_teacher_forcing = use_teacher_forcing

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, tf_target=None):
        """Long-term forecasting using ViT encoder and Transformer decoder."""
        device = next(self.parameters()).device
        batch_size = x_enc.size(0)
        
        # Step 2: Generate spectrograms from input context
        spectra_list = []
        for item in x_enc:
            spectra = get_STFT_spectra(item, device=device)
            spectra_list.append(spectra)
        
        # Stack into batch tensor
        spectra_tensor = torch.stack(spectra_list, dim=0)  # (batch, features, 128, 128)
        
        # Step 3: Process through ViT encoder
        vit_features = self.vit_encoder.get_last_hidden_state(spectra_tensor)  # (batch, num_patches+1, 768)
        encoder_features = self.encoder_projection(vit_features)  # (batch, num_patches+1, 128)

        context_condition = encoder_features[:, 0, :].unsqueeze(1)  # (batch, first patch ([CLS]), 128)
        
        # Step 4: Process through decoder with teacher forcing control
        if self.use_teacher_forcing:
            # Teacher forcing: prepare target features
            if tf_target is not None:
                # Single variable: use last feature only
                target_features = tf_target[:, :, -1:]  # (batch, pred_len, 1)
            else:
                raise ValueError("tf_target must be provided in training mode")
            
            predictions = self.ts_decoder(
                encoder_output=encoder_features,
                context_condition=context_condition,
                target=target_features,
                use_teacher_forcing=True
            )
        else:
            # Inference mode: autoregressive generation
            predictions = self.ts_decoder(
                encoder_output=encoder_features,
                context_condition=context_condition,
                target=None,
                use_teacher_forcing=False
            )
        
        # DEBUG: Focus on prediction-target learning signal
        if tf_target is not None and self.use_teacher_forcing:
            target_features = tf_target[:, :, -1:]  # Same target used in decoder
            pred_target_mse = torch.nn.functional.mse_loss(predictions, target_features)
            pred_target_mae = torch.nn.functional.l1_loss(predictions, target_features)
            
            # DEBUG: Analyze first prediction accuracy specifically
            first_pred_error = torch.nn.functional.mse_loss(predictions[:, 0:1, :], target_features[:, 0:1, :])
            print(f"[DEBUG LEARNING] First prediction MSE: {first_pred_error.item():.6f}")
            print(f"[DEBUG LEARNING] Overall MSE: {pred_target_mse.item():.6f}")
            print(f"[DEBUG LEARNING] Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}], mean: {predictions.mean():.4f}")
            print(f"[DEBUG LEARNING] Target range: [{target_features.min():.4f}, {target_features.max():.4f}], mean: {target_features.mean():.4f}")
        else:
            print(f"[DEBUG LEARNING] Inference prediction range: [{predictions.min():.4f}, {predictions.max():.4f}], mean: {predictions.mean():.4f}")
        
        return predictions

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, tf_target):
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
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, tf_target)
            return dec_out  # [B, pred_len, 1] - univariate output (last feature)
        else:
            raise ValueError(f"Task {self.task_name} not supported. Only 'long_term_forecast' and 'classification' are supported.")