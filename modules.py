import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
from torch.autograd import Function


class SpecModel(nn.Module):
  def __init__(self, sr, n_fft, hop_length, n_mels):
    super().__init__()
    self.mel_converter = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    self.db_converter = torchaudio.transforms.AmplitudeToDB()
  
  def forward(self, x):
    mel_spec = self.mel_converter(x)
    return self.db_converter(mel_spec)

class Conv_1d(nn.Module):
  def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
    super(Conv_1d, self).__init__()
    self.conv = nn.Conv1d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
    self.bn = nn.BatchNorm1d(output_channels)
    self.relu = nn.ReLU()
    self.mp = nn.MaxPool1d(pooling)
  def forward(self, x):
    x = x.float()
    out = self.mp(self.relu(self.bn(self.conv(x))))
    return out


class Conv_2d(nn.Module):
  def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
    super(Conv_2d, self).__init__()
    self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
    self.bn = nn.BatchNorm2d(output_channels)
    self.relu = nn.ReLU()
    self.mp = nn.MaxPool2d(pooling)

  def forward(self, x):
    out = self.mp(self.relu(self.bn(self.conv(x))))
    return out
    
      
class Res_2d(nn.Module):
  def __init__(self, input_channels, output_channels, shape=3, stride=2):
    super(Res_2d, self).__init__()
    # convolution
    self.conv_1 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
    self.bn_1 = nn.BatchNorm2d(output_channels)
    self.conv_2 = nn.Conv2d(output_channels, output_channels, shape, padding=shape//2)
    self.bn_2 = nn.BatchNorm2d(output_channels)

    # residual
    self.diff = False
    if (stride != 1) or (input_channels != output_channels):
        self.conv_3 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn_3 = nn.BatchNorm2d(output_channels)
        self.diff = True
    self.relu = nn.ReLU()

  def forward(self, x):
    # convolution
    out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))

    # residual
    if self.diff:
        x = self.bn_3(self.conv_3(x))
    out = x + out
    out = self.relu(out)
    return out


class ClassificationHead(nn.Module):
  def __init__(self, in_features, n_class, n_channels=1):
    super(ClassificationHead, self).__init__()
    multiplier = 3 if n_channels == 2 else 1  # distinguish the stereo case from the mono case
    self.whole_final_layer = nn.Linear(in_features * multiplier, n_class)
    self.half_final_layer = nn.Linear(in_features * multiplier, n_class*2)
    self.quarter_final_layer = nn.Linear(in_features * multiplier, n_class*4)
    self.one_year_final_layer = nn.Linear(in_features * multiplier, 2024-1958+1)
    
    self.domain_classification_layer = nn.Sequential(
        GradientReversal(1.0),
        nn.Linear(in_features * multiplier, in_features * multiplier),
        nn.ReLU(),
        nn.Linear(in_features * multiplier, 1)
    )
  
  def forward(self, x):
    whole_out = self.whole_final_layer(x).softmax(dim=-1)
    half_out = self.half_final_layer(x).softmax(dim=-1)
    quarter_out = self.quarter_final_layer(x).softmax(dim=-1)
    one_year_out = self.one_year_final_layer(x).softmax(dim=-1)
    
    domain_out = self.domain_classification_layer(x).sigmoid()
    return [whole_out, half_out, quarter_out, one_year_out, domain_out]
    
      
class Conv_V(nn.Module):
  # vertical convolution
  def __init__(self, input_channels, output_channels, filter_shape):
    super(Conv_V, self).__init__()
    self.conv = nn.Conv2d(input_channels, output_channels, filter_shape,
                          padding=(0, filter_shape[1]//2))
    self.bn = nn.BatchNorm2d(output_channels)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.bn(self.conv(x)))
    freq = x.size(2)
    out = nn.MaxPool2d((freq, 1), stride=(freq, 1))(x)
    out = out.squeeze(2)
    return out


class Conv_H(nn.Module):
  # horizontal convolution
  def __init__(self, input_channels, output_channels, filter_length):
    super(Conv_H, self).__init__()
    self.conv = nn.Conv1d(input_channels, output_channels, filter_length,
                          padding=filter_length//2)
    self.bn = nn.BatchNorm1d(output_channels)
    self.relu = nn.ReLU()

  def forward(self, x):
    freq = x.size(2)
    out = nn.AvgPool2d((freq, 1), stride=(freq, 1))(x)
    out = out.squeeze(2)
    out = self.relu(self.bn(self.conv(out)))
    return out
  
  
class ResSE_1d(nn.Module):
  def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=3):
    super(ResSE_1d, self).__init__()
    # convolution
    self.conv_1 = nn.Conv1d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
    self.bn_1 = nn.BatchNorm1d(output_channels)
    self.conv_2 = nn.Conv1d(output_channels, output_channels, shape, padding=shape//2)
    self.bn_2 = nn.BatchNorm1d(output_channels)

    # squeeze & excitation
    self.dense1 = nn.Linear(output_channels, output_channels)
    self.dense2 = nn.Linear(output_channels, output_channels)

    # residual
    self.diff = False
    if (stride != 1) or (input_channels != output_channels):
      self.conv_3 = nn.Conv1d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
      self.bn_3 = nn.BatchNorm1d(output_channels)
      self.diff = True
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.mp = nn.MaxPool1d(pooling)

  def forward(self, x):
    x = x.float()
    # convolution
    out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))

    # squeeze & excitation
    se_out = nn.AvgPool1d(out.size(-1))(out)
    se_out = se_out.squeeze(-1)
    se_out = self.relu(self.dense1(se_out))
    se_out = self.sigmoid(self.dense2(se_out))
    se_out = se_out.unsqueeze(-1)
    out = torch.mul(out, se_out)

    # residual
    if self.diff:
      x = self.bn_3(self.conv_3(x))
    out = x + out
    out = self.mp(self.relu(out))
    return out


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None
revgrad = GradientReversal.apply

class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)