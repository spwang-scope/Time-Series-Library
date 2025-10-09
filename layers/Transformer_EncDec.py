import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.resweight = nn.Parameter(torch.Tensor([0]))

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            x_mask,
            tau=tau, delta=None
        )[0]) * self.resweight

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            cross_mask,
            tau=tau, delta=delta
        )[0]) * self.resweight

        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class AutoregressiveDecoderLayer(nn.Module):
    """
    Autoregressive Decoder Layer for Time Series Forecasting
    
    Generates predictions one by one in an autoregressive manner, maintaining
    key-value caches for efficient sequential generation.
    """
    
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(AutoregressiveDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        
        # Feedforward network
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.resweight = nn.Parameter(torch.Tensor([0]))
        
    def forward_step(self, x, cross, key_cache=None, value_cache=None, 
                     cross_mask=None, tau=None, delta=None):
        """
        Forward pass for a single autoregressive step
        
        Args:
            x: Current input token [batch, 1, d_model]
            cross: Memory/encoder output for cross-attention [batch, seq_len, d_model]
            key_cache: Accumulated keys for self-attention [batch, prev_len, n_heads, d_k]
            value_cache: Accumulated values for self-attention [batch, prev_len, n_heads, d_v]
            cross_mask: Mask for cross-attention
            tau: Optional parameter for attention
            delta: Optional parameter for attention
            
        Returns:
            output: Generated token [batch, 1, d_model]
            new_key_cache: Updated key cache
            new_value_cache: Updated value cache
        """
        batch_size, seq_len, _ = x.shape
        
        # Prepare queries, keys, values for self-attention
        # Assume self_attention has linear layers q_proj, k_proj, v_proj
        #if hasattr(self.self_attention, 'query_projection'):
        #    q = self.self_attention.query_projection(x)
        #    k = self.self_attention.key_projection(x)
        #    v = self.self_attention.value_projection(x)
        #else:
        #    # Fallback for different attention implementations
        q = k = v = x
            
        # Concatenate with cached keys and values
        if key_cache is not None and value_cache is not None:
            k = torch.cat([key_cache, k], dim=1)
            v = torch.cat([value_cache, v], dim=1)
            
        # Update cache for next step
        new_key_cache = k
        new_value_cache = v
        
        # Self-attention with causal mask
        attn_mask = None
        if k.size(1) > 1:
            # Create causal mask for current position
            seq_len_total = k.size(1)
            attn_mask = torch.triu(torch.ones(seq_len_total, seq_len_total, 
                                            device=x.device, dtype=torch.bool), diagonal=1)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            attn_mask = attn_mask.expand(batch_size, -1, -1, -1)
            
        # Apply self-attention
        self_attn_out, _ = self.self_attention(q, k, v, attn_mask, 
                                             tau=tau, delta=None)
        
        # Residual connection and layer norm
        x = x + self.dropout(self_attn_out) * self.resweight
        x = self.norm1(x)
        
        # Cross-attention
        cross_attn_out, _ = self.cross_attention(x, cross, cross,
                                               cross_mask,
                                               tau=tau, delta=delta)
        
        # Residual connection and layer norm
        x = x + self.dropout(cross_attn_out) * self.resweight
        y = x = self.norm2(x)
        
        # Feedforward network
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        # Final residual connection and layer norm
        output = self.norm3(x + y)
        
        return output, new_key_cache, new_value_cache


class AutoregressiveDecoder(nn.Module):
    """
    Autoregressive Decoder for Time Series Forecasting
    
    Generates predictions sequentially using multiple decoder layers with key-value caching.
    """
    
    def __init__(self, layers, norm_layer=None, projection=None):
        super(AutoregressiveDecoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        
    def forward(self, context, memory, pred_len, cross_mask=None, tau=None, delta=None):
        """
        Autoregressive forward pass
        
        Args:
            context: Context sequence [batch, seq_len, d_model]
            memory: Memory for cross-attention [batch, memory_len, d_model]
            pred_len: Number of predictions to generate
            cross_mask: Mask for cross-attention
            tau: Optional parameter for attention
            delta: Optional parameter for attention
            
        Returns:
            predictions: Generated sequence [batch, pred_len, d_model]
        """
        batch_size, seq_len, d_model = context.shape
        device = context.device
        
        # Initialize with the last token of context as the starting point
        current_input = context[:, -1:, :]  # [batch, 1, d_model]
        predictions = []
        
        # Initialize key-value caches for each layer
        key_caches = [None] * len(self.layers)
        value_caches = [None] * len(self.layers)
        
        # Optionally include context in initial cache
        # Pre-fill cache with context for better initialization
        if seq_len > 1:
            # Process context through all layers to initialize caches
            x = context
            for i, layer in enumerate(self.layers):
                # Get keys and values from context
                #if hasattr(layer.self_attention, 'query_projection'):
                #    k_context = layer.self_attention.key_projection(x)
                #    v_context = layer.self_attention.value_projection(x)
                #else:
                k_context = v_context = x
                
                key_caches[i] = k_context
                value_caches[i] = v_context
                
                # Forward through layer (just to maintain consistency)
                x = x + layer.dropout(layer.self_attention(x, x, x, None)[0]) * layer.resweight
                x = layer.norm1(x)
                x = x + layer.dropout(layer.cross_attention(x, memory, memory, 
                                                          cross_mask,
                                                          tau=tau, delta=delta)[0]) * layer.resweight
                y = x = layer.norm2(x)
                y = layer.dropout(layer.activation(layer.conv1(y.transpose(-1, 1))))
                y = layer.dropout(layer.conv2(y).transpose(-1, 1))
                x = layer.norm3(x + y)
            
            # Use last token of processed context as starting input
            current_input = x[:, -1:, :]
        
        # Generate predictions autoregressively
        for step in range(pred_len):
            x = current_input
            
            # Pass through each decoder layer
            for i, layer in enumerate(self.layers):
                x, key_caches[i], value_caches[i] = layer.forward_step(
                    x, memory, key_caches[i], value_caches[i],
                    cross_mask=cross_mask, tau=tau, delta=delta
                )
            
            # Apply final normalization
            if self.norm is not None:
                x = self.norm(x)
                
            # Apply projection if provided
            if self.projection is not None:
                x = self.projection(x)
            
            # Store prediction and use as next input
            predictions.append(x)
            current_input = x
            
        # Concatenate all predictions
        predictions = torch.cat(predictions, dim=1)  # [batch, pred_len, d_model]
        
        return predictions


class CachedAutoregressiveDecoder(nn.Module):
    """
    Memory-efficient autoregressive decoder that uses proper key-value caching
    similar to modern transformer implementations.
    """
    
    def __init__(self, layers, norm_layer=None, projection=None):
        super(CachedAutoregressiveDecoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        
    def forward(self, context, memory, pred_len, cross_mask=None, tau=None, delta=None):
        """
        Efficient autoregressive generation with proper caching
        
        Args:
            context: Context sequence [batch, seq_len, d_model]  
            memory: Memory for cross-attention [batch, memory_len, d_model]
            pred_len: Number of predictions to generate
            cross_mask: Mask for cross-attention
            tau: Optional parameter for attention
            delta: Optional parameter for attention
            
        Returns:
            predictions: Generated sequence [batch, pred_len, d_model]
        """
        batch_size, seq_len, d_model = context.shape
        device = context.device
        
        # Start with last token of context
        generated_sequence = context[:, -1:, :]  # [batch, 1, d_model]
        all_outputs = []
        
        # Initialize past key-value states for each layer
        past_key_values = [None] * len(self.layers)
        
        for step in range(pred_len):
            # Current input is the last generated token
            current_input = generated_sequence[:, -1:, :]
            
            # Pass through each layer with caching
            x = current_input
            new_past_key_values = []
            
            for i, layer in enumerate(self.layers):
                # Get past key-values for this layer
                past_kv = past_key_values[i]
                
                # Forward with caching
                x, new_kv = self._forward_layer_with_cache(
                    layer, x, memory, past_kv, cross_mask, tau, delta
                )
                new_past_key_values.append(new_kv)
            
            # Update past key-values
            past_key_values = new_past_key_values
            
            # Apply final transformations
            if self.norm is not None:
                x = self.norm(x)
            if self.projection is not None:
                x = self.projection(x)
                
            # Append to sequence
            all_outputs.append(x)
            generated_sequence = torch.cat([generated_sequence, x], dim=1)
            
        return torch.cat(all_outputs, dim=1)
    
    def _forward_layer_with_cache(self, layer, x, memory, past_kv, cross_mask, tau, delta):
        """
        Forward pass through a single layer with key-value caching
        """
        # Self-attention with cache
        if hasattr(layer.self_attention, 'query_projection'):
            q = layer.self_attention.query_projection(x)
            k = layer.self_attention.key_projection(x) 
            v = layer.self_attention.value_projection(x)
        else:
            q = k = v = x
            
        # Concatenate with past keys/values
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
            
        # Store new key-value state
        new_kv = (k, v)
        
        # Apply self-attention (only attend to past + current)
        self_attn_out, _ = layer.self_attention(q, k, v, None, tau=tau, delta=None)
        x = x + layer.dropout(self_attn_out) * layer.resweight
        x = layer.norm1(x)
        
        # Cross-attention (no caching needed as memory is static)
        cross_attn_out, _ = layer.cross_attention(x, memory, memory,
                                                cross_mask,
                                                tau=tau, delta=delta)
        x = x + layer.dropout(cross_attn_out) * layer.resweight  
        y = x = layer.norm2(x)
        
        # Feedforward
        y = layer.dropout(layer.activation(layer.conv1(y.transpose(-1, 1))))
        y = layer.dropout(layer.conv2(y).transpose(-1, 1))
        x = layer.norm3(x + y)
        
        return x, new_kv
