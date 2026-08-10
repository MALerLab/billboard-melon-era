import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from pathlib import Path
import random
import json
import pandas as pd



RANDOM_SEED = 77
random.seed(RANDOM_SEED)

class BillboardDataset:
  def __init__(self, dir_path, df_path, json_path, sr, channel, clip_len, mode, data_type, equalize):
    self.dir = Path(dir_path)
    self.json_path = Path(json_path)
    self.sr = sr
    self.channel = channel
    self.clip_len = clip_len
    self.mode = mode
    self.decades = {1960: 0, 1970: 1, 1980: 2, 1990: 3, 2000: 4, 2010: 5}
    self.df_path = df_path
    self.dataset_df = pd.read_csv(self.df_path)
    self.whole, self.half, self.quarter, self.one_year = self._initialize_class_dicts()
    self.class_names = [list(classes.keys()) for classes in [self.whole, self.half, self.quarter, self.one_year]]
    self.data_type = data_type
    self.equalize = equalize
    # Keep the full split in audio_files even when equalizing. The decade-balanced
    # subset is re-drawn once per epoch (see BillboardDatasetHierarchyTrain), which is
    # what the paper describes; picking it once here would freeze it for the whole run.
    self.audio_files = self._load_audio_files()
    self.decade_index = self._build_decade_index() if (self.mode == 'train' and self.equalize) else None

  def _initialize_class_dicts(self):
    whole = {'WHOLE_1960':0, 'WHOLE_1970':1, 'WHOLE_1980':2, 'WHOLE_1990':3, 'WHOLE_2000':4, 'WHOLE_2010':5}
    half = {'HALF_1960_1':0, 'HALF_1960_2':1, 'HALF_1970_1':2, 'HALF_1970_2':3, 'HALF_1980_1':4, 'HALF_1980_2':5, 
            'HALF_1990_1':6, 'HALF_1990_2':7, 'HALF_2000_1':8, 'HALF_2000_2':9, 'HALF_2010_1':10, 'HALF_2010_2':11}
    quarter = {'QUARTER_1960_1':0, 'QUARTER_1960_2':1, 'QUARTER_1960_3':2, 'QUARTER_1960_4':3,
              'QUARTER_1970_1':4, 'QUARTER_1970_2':5, 'QUARTER_1970_3':6, 'QUARTER_1970_4':7,
              'QUARTER_1980_1':8, 'QUARTER_1980_2':9, 'QUARTER_1980_3':10, 'QUARTER_1980_4':11,
              'QUARTER_1990_1':12, 'QUARTER_1990_2':13, 'QUARTER_1990_3':14, 'QUARTER_1990_4':15,
              'QUARTER_2000_1':16, 'QUARTER_2000_2':17, 'QUARTER_2000_3':18, 'QUARTER_2000_4':19,
              'QUARTER_2010_1':20, 'QUARTER_2010_2':21, 'QUARTER_2010_3':22, 'QUARTER_2010_4':23}
    one_year = {str(i): i-1958 for i in range(1958, 2025)}  
    return whole, half, quarter, one_year

  @property
  def hierarchy_class_map(self):
    return [{i: [i*2, i*2+1] for i in range(6)}, # whole to half
            {i: [i*2, i*2+1] for i in range(12)}, # half to quarter
            # quarter to one year 1960
          { 0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8], 3: [9, 10, 11],
            # quarter to one year 1970~2000
            4: [12, 13, 14], 5: [15, 16], 6: [17, 18, 19], 7: [20, 21],
            8: [22, 23, 24], 9: [25, 26], 10: [27, 28, 29], 11: [30, 31],
            12: [32, 33, 34], 13: [35, 36], 14: [37, 38, 39], 15: [40, 41],
            16: [42, 43, 44], 17: [45, 46], 18: [47, 48, 49], 19: [50, 51],
            # quarter to one year 2010
            20: [52, 53, 54, 55], 21: [56, 57, 58], 22: [59, 60, 61, 62], 23: [63, 64, 65, 66]}
            ]

  def _load_audio_files(self):
    file_format = '.mp3' if self.data_type == 'billboard' else '.wav'
    with open(self.json_path, 'r') as f:
      file_list = json.load(f)[self.mode]
    return [self.dir/f'{file}{file_format}' for file in file_list]
  
  def _load_equal_audio_files(self, seed):
    # np.random.seed(seed)
    decade_files = {decade: [] for decade in range(len(self.whole.keys()))}
    file_format = '.mp3' if self.data_type == 'billboard' else '.wav'
    with open(self.json_path, 'r') as f:
      file_list = json.load(f)[self.mode]
    for file in tqdm(file_list):
      year = int(file.split('_')[0][1:-1])
      decade = self._get_whole(year)
      decade_files[decade].append(file)
    min_samples = min(len(files) for files in decade_files.values())
    sampled_files = []
    for files in decade_files.values():
      sampled_files.extend(np.random.choice(files, min_samples, replace=False))
    return [self.dir/f'{file}{file_format}' for file in sampled_files]
  
  def _build_decade_index(self):
    """Group positions in self.audio_files by decade, for per-epoch class balancing."""
    index = {decade: [] for decade in range(len(self.whole))}
    for i, path in enumerate(self.audio_files):
      year = int(path.name.split('_')[0][1:-1])
      index[self._get_whole(year)].append(i)
    return {d: idxs for d, idxs in index.items() if idxs}

  def _extract_year_from_date(self, date_value):
    if pd.api.types.is_integer_dtype(type(date_value)):
      return date_value
    return int(date_value.split('/')[-1])
  
  def _find_song_year(self, song_name, artist):
    filtered_df = self.dataset_df[(self.dataset_df['Song'].str.replace('/', ' ') == song_name) & 
                                  (self.dataset_df['Artist'].str.replace('/', ' ') == artist)]
    if not filtered_df.empty:
      return self._extract_year_from_date(filtered_df.iloc[0]['Date'])
    return None
    
  def hierarchical_label(self, file_name):
    parts = file_name.split('_')
    song_name, artist = parts[1], parts[2].removesuffix('.mp3' if self.data_type == 'billboard' else '.wav')
    song_name, artist = song_name[1:-1], artist[1:-1]

    year = self._find_song_year(song_name, artist)
    if year is None:
      print(f'Year not found for Song name: {song_name}, Artist: {artist}.')
      return None

    return (
        self._get_whole(year),
        self._get_half(year),
        self._get_quarter(year),
        self._get_one_year(year)
    )

  def _get_whole(self, year):
    if year < 1960:
      return self.whole['WHOLE_1960']
    elif year >= 2020:
      return self.whole['WHOLE_2010']
    else:
      return self.whole[f'WHOLE_{year - (year % 10)}']
  
  def _get_half(self, year):
    if year < 1970:
      if year < 1964:
        return self.half['HALF_1960_1']
      else:
        return self.half['HALF_1960_2']
    elif year >= 2010:
      if year < 2017:
        return self.half['HALF_2010_1']               
      else:
        return self.half['HALF_2010_2']
    else:
      if year % 10 < 5:
        return self.half[f'HALF_{year - (year % 10)}_1']
      else:
        return self.half[f'HALF_{year - (year % 10)}_2']
      
  def _get_quarter(self, year):
    if year < 1970:
      if year < 1964:
        if year < 1961:
          return self.quarter['QUARTER_1960_1']
        else:
          return self.quarter['QUARTER_1960_2']
      else:
        if year < 1967:
          return self.quarter['QUARTER_1960_3']
        else:
          return self.quarter['QUARTER_1960_4']
    elif year >= 2010:
      if year < 2017:
        if year < 2014:
          return self.quarter['QUARTER_2010_1']
        else:
          return self.quarter['QUARTER_2010_2']
      else:
        if year < 2021:
          return self.quarter['QUARTER_2010_3']
        else:
          return self.quarter['QUARTER_2010_4']
    else:
      if year % 10 < 5:
        if year % 10 < 3:
          return self.quarter[f'QUARTER_{year - (year % 10)}_1']
        else:
          return self.quarter[f'QUARTER_{year - (year % 10)}_2']
      else:
        if year % 10 < 8:
          return self.quarter[f'QUARTER_{year - (year % 10)}_3']
        else:
          return self.quarter[f'QUARTER_{year - (year % 10)}_4']
      
  def _get_one_year(self, year):
    return self.one_year[str(year)]
  
  def __len__(self):
    return len(self.audio_files)


class BillboardDatasetHierarchyTrain(BillboardDataset):
  def __init__(self, dir_path, df_path, json_path, sr, channel, clip_len, chunk_indices, mode, data_type, equalize=False, adv_path='csv/kpop_adversarial_audio_files.json'):
    super().__init__(dir_path, df_path, json_path, sr, channel, clip_len, mode, data_type, equalize)  
    self.mode = mode
    self.adv_path = adv_path
    self.clipped_samples = self.sr * self.clip_len
    if chunk_indices is None: # In case of adversarial training
      self.audio_files_chunk = self.adversarial_audio_files()
      self.decade_index = None
    else:
      self.audio_files_chunk = self.audio_files
    self.load_audio()
    self.resample_epoch()

  def load_audio(self):
    """Hold each track whole. The 30-second crop is drawn per access in __getitem__,
    so the remainder of the track is not dead weight."""
    self.loaded_files = []
    for file in tqdm(self.audio_files_chunk):
      pt_path = self.dir/f'pt_files/{self.channel}_{self.sr}/{file.name}.pt'
      audio = torch.load(pt_path)
      whole, half, quarter, one_year = self.hierarchical_label(file.name)
      self.loaded_files.append((audio, [whole, half, quarter, one_year]))

    return self.loaded_files

  def resample_epoch(self):
    """Re-draw the decade-balanced subset. Called once per epoch so that, as the paper
    describes, the undersampling is re-randomized rather than fixed for the whole run."""
    if self.decade_index is None:
      self.epoch_indices = list(range(len(self.loaded_files)))
      return
    per_decade = min(len(idxs) for idxs in self.decade_index.values())
    self.epoch_indices = []
    for idxs in self.decade_index.values():
      self.epoch_indices.extend(np.random.choice(idxs, per_decade, replace=False).tolist())
    random.shuffle(self.epoch_indices)

  def adversarial_audio_files(self):
    file_format = '.mp3' if self.data_type == 'billboard' else '.wav'
    with open(self.adv_path, 'r') as f:
      sampled_files = json.load(f)['Sampled Files']
    return [self.dir/f'{file}{file_format}' for file in sampled_files]

  def __len__(self):
    return len(self.epoch_indices)

  def __getitem__(self, idx):
    audio, labels = self.loaded_files[self.epoch_indices[idx]]
    max_start = audio.shape[-1] - self.clipped_samples
    start = random.randint(0, max_start) if max_start > 0 else 0
    # .clone() so the returned sample does not pin the whole track through the worker
    # queue; the resident copy in loaded_files is what stays in RAM.
    return audio[:, start:start+self.clipped_samples].clone(), labels
    

class BillboardDatasetHierarchyValidTest(BillboardDataset):
  def __init__(self, dir_path, df_path, json_path, sr, channel, clip_len, mode, data_type, equalize=False):
    super().__init__(dir_path, df_path, json_path, sr, channel, clip_len, mode, data_type, equalize)  
    self.mode = mode

    if self.mode == 'valid':
      self.save_random_30_sec_pt_files()
      self.load_audio()
    elif self.mode == 'test':
      self.testset_list = self.save_all_30_sec_pt_files()
      self.load_audio()
      
  def save_random_30_sec_pt_files(self):
    for file in tqdm(self.audio_files):
      pt_path = self.dir/f'pt_files/{self.channel}_{self.sr}/{self.mode}/{file.name}.pt'
      if pt_path.exists():
        continue
      audio, org_sr = torchaudio.load(file)
      if self.sr != 48000: # default=48000
        audio = torchaudio.functional.resample(audio, orig_freq=org_sr, new_freq=self.sr)
      if self.channel == 'mono':
        audio = audio.mean(dim=0).unsqueeze(0)
      clipped_samples = self.sr * self.clip_len
      max_start = audio.shape[-1] - clipped_samples
      start = random.randint(0, max_start-1)
      audio = audio[:, start:start+clipped_samples]
      audio = audio.to(dtype=torch.float16)
      pt_path.parent.mkdir(parents=True, exist_ok=True)
      torch.save(audio, pt_path)  
    
  def save_all_30_sec_pt_files(self): # Cut the audio into 30-second and save it as pt file. (Not Randomly)
    for file in tqdm(self.audio_files):
      pt_path = self.dir/f'pt_files/{self.channel}_{self.sr}/{self.mode}/slices_30_sec/{file.stem}_0.pt'
      if pt_path.exists():
        continue
      audio, org_sr = torchaudio.load(file)
      if self.sr != 48000: # default=48000
        audio = torchaudio.functional.resample(audio, orig_freq=org_sr, new_freq=self.sr)
      if self.channel == 'mono':
        audio = audio.mean(dim=0).unsqueeze(0)
        
      len_audio = audio.shape[-1]
      sample_length = self.sr * 30  # 30 seconds
      num_full_segments = len_audio // sample_length
      total_full_segment_length = num_full_segments * sample_length
      
      remaining_samples = len_audio - total_full_segment_length
      trim_start = remaining_samples // 2
      trim_end = remaining_samples - trim_start
      audio = audio[:, trim_start:len_audio-trim_end]
      
      for segment_index in range(num_full_segments):
        start_sample = segment_index * sample_length
        end_sample = start_sample + sample_length
        segment = audio[:, start_sample:end_sample].to(dtype=torch.float16)
        segment_save_path = pt_path.parent/f'{file.stem}_{segment_index}.pt'
        segment_save_path.parent.mkdir(parents=True, exist_ok=True)
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(segment, segment_save_path)
    
    # testset path
    testset_paths = self.dir/f'pt_files/{self.channel}_{self.sr}/{self.mode}/slices_30_sec'
    testset_list = list(testset_paths.glob('*.pt'))
    
    return testset_list
  
  def load_audio(self):
    self.loaded_files = []
    if self.mode == 'valid':
      for file in tqdm(self.audio_files):
        pt_path = self.dir/f'pt_files/{self.channel}_{self.sr}/{self.mode}/{file.name}.pt'
        audio = torch.load(pt_path)
        whole, half, quarter, one_year = self.hierarchical_label(file.name)
        self.loaded_files.append((audio, [whole, half, quarter, one_year]))
    elif self.mode == 'test':
      for file in tqdm(self.testset_list):
        audio = torch.load(file)
        whole, half, quarter, one_year = self.hierarchical_label(file.name)
        self.loaded_files.append((audio, [whole, half, quarter, one_year], file.name))
      
    return self.loaded_files
  
  def __len__(self):
    if self.mode == 'valid':
      return len(self.audio_files)
    elif self.mode == 'test':
      return len(self.testset_list)
  
  def __getitem__(self, idx):
    if self.mode == 'valid':
      audio, labels = self.loaded_files[idx]
      return audio, labels
    elif self.mode == 'test':
      audio, labels, file_name = self.loaded_files[idx]
      return audio, labels, file_name


class BillboardDatasetTrain(BillboardDataset):
  def __init__(self, dir_path, df_path, json_path, chunk_indices, sr, channel, clip_len):
    super().__init__(dir_path, df_path, json_path, sr, channel, clip_len, 'train')

    self.audio_files_chunk = [self.audio_files[i] for i in chunk_indices]
    self.load_audio()
        
  def load_audio(self):
    self.loaded_files = []
    for file in self.audio_files_chunk:
      pt_path = self.dir/f'pt_files/{self.channel}_{self.sr}/{file.name}.pt'
      audio = torch.load(pt_path)
      clipped_samples = self.sr * self.clip_len
      max_start = audio.shape[-1] - clipped_samples
      start = random.randint(0, max_start-1)
      audio = audio[:, start:start+clipped_samples]
      
      year = int(file.name.split('_')[0][1:-1])
      if year < 1960:
        label = self.decades[1960]
      elif year >= 2020:
        label = self.decades[2010]
      else:
        label = self.decades[year - (year % 10)]
      self.loaded_files.append((audio, label))
    return self.loaded_files

  def __len__(self):
    return len(self.audio_files_chunk)
  
  def __getitem__(self, idx):
    audio, label = self.loaded_files[idx]
    return audio, label
    

class BillboardDatasetValidTest(BillboardDataset):
  def __init__(self, dir_path, df_path, json_path, sr, channel, clip_len, mode):
    super().__init__(dir_path, df_path, json_path, sr, channel, clip_len, mode)

    self.save_30_sec_pt_files()
    self.loaded_files = []
    self.load_audio()
    
  def save_30_sec_pt_files(self):
    for file in tqdm(self.audio_files):
      pt_path = self.dir/f'pt_files/{self.channel}_{self.sr}/{self.mode}/{file.name}.pt'
      if pt_path.exists():
        continue
      audio, org_sr = torchaudio.load(file)
      if self.sr != 48000: # default=48000
        audio = torchaudio.functional.resample(audio, orig_freq=org_sr, new_freq=self.sr)
      if self.channel == 'mono':
        audio = audio.mean(dim=0).unsqueeze(0)
      clipped_samples = self.sr * self.clip_len
      max_start = audio.shape[-1] - clipped_samples
      start = random.randint(0, max_start-1)
      audio = audio[:, start:start+clipped_samples]
      audio = audio.to(dtype=torch.float16)
      pt_path.parent.mkdir(parents=True, exist_ok=True)
      torch.save(audio, pt_path)
    
        
  def load_audio(self):
    self.loaded_files = []
    for file in self.audio_files:
      pt_path = self.dir/f'pt_files/{self.channel}_{self.sr}/{self.mode}/{file.name}.pt'
      audio = torch.load(pt_path)
      
      year = int(file.name.split('_')[0][1:-1])
      if year < 1960:
        label = self.decades[1960]
      elif year >= 2020:
        label = self.decades[2010]
      else:
        label = self.decades[year - (year % 10)]
      self.loaded_files.append((audio, label, file.name))
    return self.loaded_files
  
  
  def __len__(self):
    return len(self.audio_files)
  
  def __getitem__(self, idx):
    audio, label, file_name = self.loaded_files[idx]
    return audio, label, file_name
  
  
import concurrent.futures


class BillboardDatasetSegment(BillboardDataset):
  def __init__(self, dir_path, df_path, json_path, sr, channel, segment_start, segment_end, mode, data_type, equalize=False):
    self.segment_start = int(segment_start)
    self.segment_end = int(segment_end)
    super().__init__(dir_path, df_path, json_path, sr, channel, self.segment_end - self.segment_start, mode, data_type, equalize)
    if self.equalize:
      self.audio_files = self._load_equal_audio_files(RANDOM_SEED)
    else:
      self.audio_files = self._load_audio_files()
    self.save_segmented_audio()
    self.load_segmented_audio()

  def _load_equal_audio_files(self, seed):
    np.random.seed(seed)
    decade_files = {decade: [] for decade in range(len(self.whole.keys()))}
    file_format = '.mp3' if self.data_type == 'billboard' else '.wav'
    with open(self.json_path, 'r') as f:
      file_list = json.load(f)[self.mode]
    for file in tqdm(file_list):
      year = int(file.split('_')[0][1:-1])
      decade = self._get_whole(year)
      decade_files[decade].append(file)
    # print({key: len(value) for key, value in decade_files.items()})
    min_samples = min(len(files) for files in decade_files.values())
    sampled_files = []
    for files in decade_files.values():
      sampled_files.extend(np.random.choice(files, min_samples, replace=False))
    return [self.dir/f'{file}{file_format}' for file in sampled_files]

  def save_segmented_audio(self):
    segment_dir = self.dir / f'segments_{self.segment_start}_{self.segment_end}'
    segment_dir.mkdir(parents=True, exist_ok=True)

    def process_and_save(file):
      pt_path = segment_dir / f'{file.stem}.pt'
      if pt_path.exists():
        return
      audio, org_sr = torchaudio.load(file)
      if self.sr != 48000:  # default=48000
        audio = torchaudio.functional.resample(audio, orig_freq=org_sr, new_freq=self.sr)
      if self.channel == 'mono':
        audio = audio.mean(dim=0).unsqueeze(0)

      start_sample = int(self.sr * self.segment_start)
      end_sample = int(self.sr * self.segment_end)
      audio_segment = audio[:, start_sample:end_sample]
      audio_segment = audio_segment.to(dtype=torch.float16)
      torch.save(audio_segment, pt_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
      list(tqdm(executor.map(process_and_save, self.audio_files), total=len(self.audio_files)))

  def load_segmented_audio(self):
    self.loaded_files = []
    segment_dir = self.dir / f'segments_{self.segment_start}_{self.segment_end}'

    def load_file(file):
      pt_path = segment_dir / f'{file.stem}.pt'
      audio_segment = torch.load(pt_path)
      whole, half, quarter, one_year = self.hierarchical_label(file.name)
      return (audio_segment, [whole, half, quarter, one_year], file.name)

    with concurrent.futures.ThreadPoolExecutor() as executor:
      results = list(tqdm(executor.map(load_file, self.audio_files), total=len(self.audio_files)))

    self.loaded_files = results

  def __len__(self):
    return len(self.audio_files)

  def __getitem__(self, idx):
    audio, labels, file_name = self.loaded_files[idx]
    return audio, labels, file_name
  