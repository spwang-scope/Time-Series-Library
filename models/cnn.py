import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
from einops.layers.torch import Rearrange
import torchvision.models as models
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
        Spectrogram tensor of shape (num_features, 64, 64)
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
        
        # Resize spectrum to 64*64 using bilinear interpolation
        spec_expanded = spec.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 32, target_width)
        spec_final = F.interpolate(
            spec_expanded,
            size=(64, 64),
            mode='bilinear',
            align_corners=False
        )
        spec_final = spec_final[0, 0]  # Direct indexing instead of squeeze
        
        spectra_list.append(spec_final)
    
    # Stack all spectra along first dimension
    spectra = torch.stack(spectra_list, dim=0)
    
    return spectra

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

'''
class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res
'''

class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        #self.model = nn.ModuleList([TimesBlock(configs)
        #                            for _ in range(configs.e_layers)])
        #self.patch_embedding = PatchEmbedding(
        #    d_model=configs.d_model, patch_len=8, stride=8, padding=0, dropout=configs.dropout)

        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)
            
        self.my_resnet = models.resnet34(weights=None,progress=False)
        self.my_resnet.conv1 = nn.Conv2d(configs.enc_in, 64, kernel_size=8, stride=8, padding=0, bias=False)

        self.my_resnet.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # 局部擴散，保持大小
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.my_resnet.layer2[0].conv1.stride = (1, 1)
        self.my_resnet.layer2[0].downsample[0].stride = (1, 1)
        self.my_resnet.layer3[0].conv1.stride = (1, 1)
        self.my_resnet.layer3[0].downsample[0].stride = (1, 1)
        self.my_resnet.layer4[0].conv1.stride = (1, 1)
        self.my_resnet.layer4[0].downsample[0].stride = (1, 1)
        self.my_resnet.avgpool = nn.AdaptiveMaxPool2d((7, 7))  # 去掉avgpool
        self.my_resnet.fc = Rearrange('b (h w c) -> b (h w) c', h=7, w=7, c=512)

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
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        #enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        #enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
        #    0, 2, 1)  # align temporal dimension
        # TimesNet
        #for i in range(self.layer):
        #    enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        device = next(self.parameters()).device
        spectra_list = []
        for item in x_enc:
            spectra = get_STFT_spectra(item, device=device)
            spectra_list.append(spectra)
        
        # Stack into batch tensor
        spectra_tensor = torch.stack(spectra_list, dim=0)  # (batch, channels, img_height, img_width)

        enc_out = self.my_resnet(spectra_tensor)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        dec_out_length = dec_out.shape[1]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, dec_out_length, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(  
                      1, dec_out_length, 1)))
        return dec_out
    '''
    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output
    '''

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        '''
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        '''
        return None
