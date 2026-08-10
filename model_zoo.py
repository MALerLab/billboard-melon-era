import torch
import torch.nn as nn
import torchaudio
from modules import Conv_1d, Conv_2d, Res_2d, ResSE_1d, Conv_V, Conv_H, SpecModel, ClassificationHead


class AudioModel(nn.Module):
  def __init__(self, sr, channel, n_fft, hop_length, n_mels, out_channels, n_class):
    super().__init__()
    self.sr = sr
    self.channel = channel
    self.n_channels = {'mono': 1, 'stereo': 2}[self.channel]
    self.spec_converter = SpecModel(sr, n_fft, hop_length, n_mels) # shape: (batch_size, channels, n_mels, time_frames)
    self.batchnorm1d = nn.BatchNorm1d(self.n_channels*n_mels)

    # model architecture
    self.conv_layer = nn.Sequential(
        nn.Conv2d(self.n_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(3),

        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), 
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(3),

        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), 
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(3),
      )
    
    # TODO: currently hard-coded.
    if self.n_channels == 2: # stereo
      self.final_layer = nn.Linear(out_channels * 3, n_class)
    else:
      self.final_layer = nn.Linear(out_channels, n_class)

  def get_spec(self, x):
    '''
    Get result of self.spec_converter
    x (torch.Tensor): audio samples (num_batch_size X num_audio_samples)
    '''
    return self.spec_converter(x)
  
  def forward(self, x, return_embedding=False):
    spec = self.get_spec(x)
    spec_reshaped = spec.view(spec.shape[0], -1, spec.shape[-1])
    out = self.batchnorm1d(spec_reshaped)
    out = out.view(spec.shape[0], spec.shape[1], spec.shape[2], spec.shape[3])
    out = self.conv_layer(spec)
    out = out.flatten(1, 2)
    out = out.mean(dim=-1)
    if return_embedding:
      return out 
    out = self.final_layer(out)
    out = out.softmax(dim=-1)
    return out
  
  
class Baseline(nn.Module):
  def __init__(self, sr, channel, n_fft, hop_length, n_mels, out_channels, n_class):
    super().__init__()
    self.sr = sr
    self.channel = channel
    self.n_channels = {'mono': 1, 'stereo': 2}[self.channel]
    self.spec_converter = SpecModel(sr, n_fft, hop_length, n_mels) # shape: (batch_size, channels, n_mels, time_frames)
    self.batchnorm1d = nn.BatchNorm1d(self.n_channels*n_mels)

    # model architecture
    self.conv_layer = nn.Sequential( 
        nn.Conv2d(self.n_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(3),

        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), 
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(3), 

        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), 
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(3), 
      )
    
    self.classification_head = ClassificationHead(out_channels, n_class, self.n_channels)

  def get_spec(self, x):
    return self.spec_converter(x)
  
  def forward(self, x, return_embedding=False):
    spec = self.get_spec(x)
    spec_reshaped = spec.view(spec.shape[0], -1, spec.shape[-1])
    x = self.batchnorm1d(spec_reshaped)
    x = x.view(spec.shape[0], spec.shape[1], spec.shape[2], spec.shape[3])
    x = self.conv_layer(spec)
    x = x.flatten(1, 2)
    x = x.mean(dim=-1)
    if return_embedding:
      return x 
    return self.classification_head(x)


class ShortChunkCNN(nn.Module):
  def __init__(self, 
              sr, 
              channel='mono', 
              out_channels=128, 
              n_fft=512, 
              hop_length=512, 
              n_mels=128, 
              n_class=6):
    super().__init__()
    
    self.spec_converter = SpecModel(sr, n_fft, hop_length, n_mels)
    self.channel = channel
    self.n_channels = {'mono': 1, 'stereo': 2}[self.channel]

    # CNN
    self.layer1 = Conv_2d(self.n_channels, out_channels, pooling=2)
    self.layer2 = Conv_2d(out_channels, out_channels, pooling=2)
    self.layer3 = Conv_2d(out_channels, out_channels*2, pooling=2)
    self.layer4 = Conv_2d(out_channels*2, out_channels*2, pooling=2)
    self.layer5 = Conv_2d(out_channels*2, out_channels*2, pooling=2)
    self.layer6 = Conv_2d(out_channels*2, out_channels*2, pooling=2)
    self.layer7 = Conv_2d(out_channels*2, out_channels*4, pooling=2)

    # Dense
    self.dense1 = nn.Linear(out_channels*4, out_channels*4)
    self.bn = nn.BatchNorm1d(out_channels*4)
    
    self.classification_head = ClassificationHead(out_channels*4, n_class, self.n_channels)

    self.dropout = nn.Dropout(0.5)
    self.relu = nn.ReLU()

  def forward(self, x, return_embedding=False):
    x = self.spec_converter(x)  # Use SpecModel for spectrogram conversion

    # CNN
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)
    x = self.layer7(x)

    x = x.squeeze(2)

    # Global Max Pooling
    if x.size(-1) != 1:
        x = nn.MaxPool1d(x.size(-1))(x)
    x = x.squeeze(2)

    # Dense
    x = self.dense1(x)
    x = self.bn(x)
    x = self.relu(x)
    x = self.dropout(x)
    if return_embedding:
      return x 
    
    return self.classification_head(x)



class ShortChunkCNN_Res(nn.Module):
  def __init__(self, 
              sr, 
              channel='mono', 
              out_channels=128, 
              n_fft=512, 
              hop_length=512, 
              n_mels=128, 
              n_class=6):
    super().__init__()
    
    self.spec_converter = SpecModel(sr, n_fft, hop_length, n_mels)
    self.channel = channel
    self.n_channels = {'mono': 1, 'stereo': 2}[self.channel]

    # CNN
    self.layer1 = Res_2d(self.n_channels, out_channels, stride=2)
    self.layer2 = Res_2d(out_channels, out_channels, stride=2)
    self.layer3 = Res_2d(out_channels, out_channels*2, stride=2)
    self.layer4 = Res_2d(out_channels*2, out_channels*2, stride=2)
    self.layer5 = Res_2d(out_channels*2, out_channels*2, stride=2)
    self.layer6 = Res_2d(out_channels*2, out_channels*2, stride=2)
    self.layer7 = Res_2d(out_channels*2, out_channels*4, stride=2)

    # Dense
    self.dense1 = nn.Linear(out_channels*4, out_channels*4)
    self.bn = nn.BatchNorm1d(out_channels*4)
      
    self.classification_head = ClassificationHead(out_channels*4, n_class, self.n_channels)

    self.dropout = nn.Dropout(0.5)
    self.relu = nn.ReLU()

  def forward(self, x):
      x = self.spec_converter(x)  # Use SpecModel for spectrogram conversion

      # CNN
      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.layer4(x)
      x = self.layer5(x)
      x = self.layer6(x)
      x = self.layer7(x)

      x = x.squeeze(2)

      # Global Max Pooling
      if x.size(-1) != 1:
          x = nn.MaxPool1d(x.size(-1))(x)
      x = x.squeeze(2)

      # Dense
      x = self.dense1(x)
      x = self.bn(x)
      x = self.relu(x)
      x = self.dropout(x)
      
      return self.classification_head(x)


class FCN(nn.Module):
    '''
    Choi et al. 2016
    Automatic tagging using deep convolutional neural networks.
    Fully convolutional network.
    '''
    def __init__(self,
                sr=16000,
                channel='mono',
                out_channels=64,
                n_fft=512,
                hop_length=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=96,
                n_class=6):
      super(FCN, self).__init__()
      # Spectrogram      
      self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                        n_fft=n_fft,
                                                        f_min=f_min,
                                                        f_max=f_max,
                                                        n_mels=n_mels)
      self.to_db = torchaudio.transforms.AmplitudeToDB()
      self.channel = channel
      self.n_channels = {'mono': 1, 'stereo': 2}[self.channel]
      self.spec_bn = nn.BatchNorm2d(1)

      # FCN
      self.layer1 = Conv_2d(self.n_channels, out_channels, pooling=(2,4))
      self.layer2 = Conv_2d(out_channels, out_channels*2, pooling=(2,4))
      self.layer3 = Conv_2d(out_channels*2, out_channels*2, pooling=(2,4))
      self.layer4 = Conv_2d(out_channels*2, out_channels*2, pooling=(3,5))
      self.layer5 = Conv_2d(out_channels*2, out_channels, pooling=(4,4))

      # Dense
      self.dense = nn.Linear(out_channels, n_class)
      self.dropout = nn.Dropout(0.5)
      
      # Hierarchical final layer
      self.classification_head = ClassificationHead(out_channels, n_class, self.n_channels)

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)

        # FCN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Dense
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        # x = self.dense(x)
        # x = nn.Sigmoid()(x)

        return self.classification_head(x)



class Musicnn(nn.Module):
  '''
  Pons et al. 2017
  End-to-end learning for music audio tagging at scale.
  This is the updated implementation of the original paper. Referred to the Musicnn code.
  https://github.com/jordipons/musicnn
  '''
  def __init__(self,
              sr=16000,
              channel='mono',
              n_fft=512,
              f_min=0.0,
              f_max=8000.0,
              n_mels=96,
              n_class=6):
    super(Musicnn, self).__init__()

    self.channel = channel
    self.n_channels = {'mono': 1, 'stereo': 2}[self.channel]

    # Spectrogram
    self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                    n_fft=n_fft,
                                                    f_min=f_min,
                                                    f_max=f_max,
                                                    n_mels=n_mels)
    self.to_db = torchaudio.transforms.AmplitudeToDB()
    self.spec_bn = nn.BatchNorm2d(1)

    # Pons front-end
    m1 = Conv_V(self.n_channels, 204, (int(0.7*96), 7))
    m2 = Conv_V(1, 204, (int(0.4*96), 7))
    m3 = Conv_H(1, 51, 129)
    m4 = Conv_H(1, 51, 65)
    m5 = Conv_H(1, 51, 33)
    self.layers = nn.ModuleList([m1, m2, m3, m4, m5])

    # Pons back-end
    backend_channel= 64 # if dataset=='msd' else 64
    self.layer1 = Conv_1d(561, backend_channel, 7, 1, 1)
    self.layer2 = Conv_1d(backend_channel, backend_channel, 7, 1, 1)
    self.layer3 = Conv_1d(backend_channel, backend_channel, 7, 1, 1)

    # Dense
    dense_channel = 200 # if dataset=='msd' else 200
    self.dense1 = nn.Linear((561+(backend_channel*3))*2, dense_channel)
    self.bn = nn.BatchNorm1d(dense_channel)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.5)
    # self.dense2 = nn.Linear(dense_channel, n_class)
    self.classification_head = ClassificationHead(dense_channel, n_class, self.n_channels)

  def forward(self, x):
    # Spectrogram
    x = self.spec(x)
    x = self.to_db(x)
    x = self.spec_bn(x)

    # Pons front-end
    out = []
    for layer in self.layers:
      out.append(layer(x))
    out = torch.cat(out, dim=1)

    # Pons back-end
    length = out.size(2)
    res1 = self.layer1(out)
    res2 = self.layer2(res1) + res1
    res3 = self.layer3(res2) + res2
    out = torch.cat([out, res1, res2, res3], 1)

    mp = nn.MaxPool1d(length)(out)
    avgp = nn.AvgPool1d(length)(out)

    out = torch.cat([mp, avgp], dim=1)
    out = out.squeeze(2)

    out = self.relu(self.bn(self.dense1(out)))
    out = self.dropout(out)
    # out = self.dense2(out)
    # out = nn.Sigmoid()(out)

    return self.classification_head(out)


class CRNN(nn.Module):
  '''
  Choi et al. 2017
  Convolution recurrent neural networks for music classification.
  Feature extraction with CNN + temporal summary with RNN
  '''
  def __init__(self,
              sr=16000,
              channel='mono',
              n_fft=512,
              f_min=0.0,
              f_max=8000.0,
              n_mels=96,
              n_class=6):
    super(CRNN, self).__init__()
    self.channel = channel
    self.n_channels = {'mono': 1, 'stereo': 2}[self.channel]

    # Spectrogram
    self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                    n_fft=n_fft,
                                                    f_min=f_min,
                                                    f_max=f_max,
                                                    n_mels=n_mels)
    self.to_db = torchaudio.transforms.AmplitudeToDB()
    self.spec_bn = nn.BatchNorm2d(1)

    # CNN
    self.layer1 = Conv_2d(self.n_channels, 64, pooling=(2,2))
    self.layer2 = Conv_2d(64, 128, pooling=(3,3))
    self.layer3 = Conv_2d(128, 128, pooling=(4,4))
    self.layer4 = Conv_2d(128, 128, pooling=(4,4))

    # RNN
    self.layer5 = nn.GRU(128, 32, 2, batch_first=True)

    # Dense
    self.dropout = nn.Dropout(0.5)
    # self.dense = nn.Linear(32, 50)
    self.classification_head = ClassificationHead(32, n_class, self.n_channels)

  def forward(self, x):
    # Spectrogram
    x = self.spec(x)
    x = self.to_db(x)
    x = self.spec_bn(x)

    # CCN
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    # RNN
    x = x.squeeze(2)
    x = x.permute(0, 2, 1)
    x, _ = self.layer5(x)
    x = x[:, -1, :]

    # Dense
    x = self.dropout(x)
    # x = self.dense(x)
    # x = nn.Sigmoid()(x)

    return self.classification_head(x)
  
  
class SampleCNN(nn.Module):
  '''
  Lee et al. 2017
  Sample-level deep convolutional neural networks for music auto-tagging using raw waveforms.
  Sample-level CNN.
  '''
  def __init__(self,
              n_class=6):
    super(SampleCNN, self).__init__()
    self.layer1 = Conv_1d(1, 128, shape=3, stride=3, pooling=1)
    self.layer2 = Conv_1d(128, 128, shape=3, stride=1, pooling=3)
    self.layer3 = Conv_1d(128, 128, shape=3, stride=1, pooling=3)
    self.layer4 = Conv_1d(128, 256, shape=3, stride=1, pooling=3)
    self.layer5 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
    self.layer6 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
    self.layer7 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
    self.layer8 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
    self.layer9 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
    self.layer10 = Conv_1d(256, 512, shape=3, stride=1, pooling=3)
    self.layer11 = Conv_1d(512, 512, shape=1, stride=1, pooling=1)
    self.dropout = nn.Dropout(0.5)
    # self.dense = nn.Linear(512, n_class)
    self.classification_head = ClassificationHead(512, n_class, 1)

  def forward(self, x):
    # x = x.unsqueeze(1)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)
    x = self.layer7(x)
    x = self.layer8(x)
    x = self.layer9(x)
    x = self.layer10(x)
    x = self.layer11(x)
    x = x.squeeze(-1)
    x = self.dropout(x)
    # x = self.dense(x)
    # x = nn.Sigmoid()(x)
    return self.classification_head(x)
  
  
class SampleCNNSE(nn.Module):
  '''
  Kim et al. 2018
  Sample-level CNN architectures for music auto-tagging using raw waveforms.
  Sample-level CNN + residual connections + squeeze & excitation.
  '''
  def __init__(self,
              n_class=6):
    super(SampleCNNSE, self).__init__()
    self.layer1 = ResSE_1d(1, 128, shape=3, stride=3, pooling=1)
    self.layer2 = ResSE_1d(128, 128, shape=3, stride=1, pooling=3)
    self.layer3 = ResSE_1d(128, 128, shape=3, stride=1, pooling=3)
    self.layer4 = ResSE_1d(128, 256, shape=3, stride=1, pooling=3)
    self.layer5 = ResSE_1d(256, 256, shape=3, stride=1, pooling=3)
    self.layer6 = ResSE_1d(256, 256, shape=3, stride=1, pooling=3)
    self.layer7 = ResSE_1d(256, 256, shape=3, stride=1, pooling=3)
    self.layer8 = ResSE_1d(256, 256, shape=3, stride=1, pooling=3)
    self.layer9 = ResSE_1d(256, 256, shape=3, stride=1, pooling=3)
    self.layer10 = ResSE_1d(256, 512, shape=3, stride=1, pooling=3)
    self.layer11 = ResSE_1d(512, 512, shape=1, stride=1, pooling=1)
    self.dropout = nn.Dropout(0.5)
    self.dense1 = nn.Linear(512, 512)
    self.bn = nn.BatchNorm1d(512)
    # self.dense2 = nn.Linear(512, n_class)
    self.classification_head = ClassificationHead(512, n_class, 1)

  def forward(self, x):
    # x = x.unsqueeze(1)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)
    x = self.layer7(x)
    x = self.layer8(x)
    x = self.layer9(x)
    x = self.layer10(x)
    x = self.layer11(x)
    x = x.squeeze(-1)
    x = nn.ReLU()(self.bn(self.dense1(x)))
    x = self.dropout(x)
    # x = self.dense2(x)
    # x = nn.Sigmoid()(x)
    return self.classification_head(x)
