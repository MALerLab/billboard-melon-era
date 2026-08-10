import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import IPython.display as ipd
from datetime import datetime
import os
from torch.utils.data import Dataset
import random
  
DEV = 'cuda' # select your device 'cpu' or 'cuda'

class PreprocessData():
  def __init__(self, sr, channel, audio_path='data', out_path='data/pt_files'):
    self.audio_dir = Path(audio_path)
    self.audio_files = list(self.audio_dir.glob('*.mp3')) + list(self.audio_dir.glob('*.wav'))
    self.out_dir = Path(out_path)
    self.sr = sr
    self.channel = channel
    self.save_pt_file()
        
  def save_pt_file(self):
    no_stereo = []
    for file in tqdm(self.audio_files):
      pt_path = self.out_dir/f'{self.channel}_{self.sr}/{file.name}.pt'
      pt_path.parent.mkdir(parents=True, exist_ok=True)
      # if pt_path.exists():
      #   continue
      audio, org_sr = torchaudio.load(file)
      if self.sr != 48000: # default=48000
        audio = torchaudio.functional.resample(audio, orig_freq=org_sr, new_freq=self.sr)
      if self.channel == 'mono':
        audio = audio.mean(dim=0).unsqueeze(0)
      if self.channel == 'stereo' and audio.shape[0] != 2: # In case of defective stereo file. Delete it later.
        no_stereo.append(file.name)
        continue
      audio = audio.to(dtype=torch.float16)
      torch.save(audio, pt_path)
    
    with open('no_stereo.txt', 'w') as f:
      for file in no_stereo:
        f.write(f'{file}\n')


if __name__ == '__main__':
  # TODO: Handle these as arguments
  channel = 'mono'
  sr = 16000
  # base_path = Path('remastered_original')
  # mp3_paths = [base_path / 'remastered', base_path / 'original']
  # for mp3_path in mp3_paths:
  #   PreprocessData(mp3_path=mp3_path, out_path=f'{mp3_path}/pt_files', sr=sr, channel=channel)
  # base_path = Path('kpop_data')
  # PreprocessData(audio_path=base_path, out_path=f'{base_path}/pt_files', sr=sr, channel=channel)
  base_path = Path('data')
  PreprocessData(audio_path=base_path, out_path=f'{base_path}/pt_files', sr=sr, channel=channel)