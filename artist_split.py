import pandas as pd
import numpy as np
import json
import random
from tqdm import tqdm

class ArtistSplit:
  def __init__(self, df_path, out_path, seed=77):
    self.csv_path = df_path
    self.seed = seed
    self.splits = [
        ' and ', '&', ' featuring ', '\\(featuring ', '\\(feat', ' ft\\.', ' feat ', ' feat\\.', 'with ',
        '\\(with ', '\\, ', ' x ', ' presents', ' \\+ ', '\\/'
    ]
    self.df = pd.read_csv(df_path, encoding_errors='ignore')
    self.artist_collaboration_dict = {}
    self.new_collaboration_dict = {}
    self.grouped_song = {}
    self.dataset = {}   
    self.out_path = out_path 
    
  def update_collaborations(self):
    updated_artists = set()

    for artist, collabs in tqdm(self.artist_collaboration_dict.items(), desc="Updating collaborations"):
      if artist not in updated_artists and collabs:
        changed = True
        while changed:
          changed = False
          new_collabs = []
          for collab in collabs:
            if collab in self.artist_collaboration_dict:
              new_collabs.extend([c for c in self.artist_collaboration_dict[collab] if c not in collabs and c not in new_collabs and c != artist])

          collabs.extend(new_collabs)
          collabs = list(set(collabs))
          updated_artists.update(collabs)
          
          if len(new_collabs) > 0:
            changed = True
        self.new_collaboration_dict[artist] = collabs
        updated_artists.add(artist)
      elif artist not in updated_artists and not collabs:
        self.new_collaboration_dict[artist] = []
        updated_artists.add(artist)
    
  def generate_songs_per_artist(self):
    for artist, collabs in tqdm(self.new_collaboration_dict.items(), desc="Grouping songs"):
      songs = set()
      collabs_with_artist = collabs + [artist]
      for collab in collabs_with_artist:
        for idx in range(self.artists_name_split.shape[1]):
          found_rows = self.sorted_df[self.sorted_df[f'Artist{idx}'] == collab]
          # {year}_{song}_{org_artist}
          songs.update([f"{{{row['Date'].split('/')[2]}}}_{{{row['Song']}}}_{{{row['Org Artist']}}}" for idx, row in found_rows.iterrows()])
      self.grouped_song[artist] = list(songs)  
    
  def split_datasets(self):
    random.seed(self.seed)
    grouped_song_keys = list(self.grouped_song.keys())
    random.shuffle(grouped_song_keys)

    total_songs = sum(len(songs) for songs in self.grouped_song.values())
    train_songs = int(total_songs * 0.8)
    valid_songs = int(total_songs * 0.1)
    test_songs = total_songs - train_songs - valid_songs
    train, valid, test = [], [], []

    for idx, key in enumerate(grouped_song_keys):
      songs = self.grouped_song[key]
      if idx % 3 == 0 and len(valid) + len(songs) <= valid_songs * 1.1:
          valid.extend(songs)
      elif idx % 3 == 1 and len(test) + len(songs) <= test_songs * 1.1:
          test.extend(songs)
      else:
          train.extend(songs)

    self.dataset = {
        "train": train,
        "valid": valid,
        "test": test
    }

  def save_to_json(self, file_path):
    with open(file_path, 'w') as file:
      json.dump(self.dataset, file, indent=4)
    

class ArtistSplitBillboard(ArtistSplit):
  def __init__(self, df_path, out_path, seed=77):
    super().__init__(df_path, out_path, seed)

  def load_and_process_data(self):
    artists_lower_stripped = self.df['Artist'].str.lower().str.strip()
    self.artists_name_split = artists_lower_stripped.str.split('|'.join(self.splits), expand=True, regex=True).applymap(lambda x: x.strip().replace(')', '') if isinstance(x, str) else x)
    
    column_names = ['Date', 'Song', 'Org Artist'] + ['Artist' + str(i) for i in range(self.artists_name_split.shape[1])]
    self.df['Song'] = self.df['Song'].str.replace('/', ' ')
    self.df['Artist'] = self.df['Artist'].str.replace('/', ' ')

    self.sorted_df = pd.concat([self.df[['Date', 'Song', 'Artist']], self.artists_name_split], axis=1)
    self.sorted_df.columns = column_names

  def remove_his_list(self):
    his_list = ['band', 'chorus', 'orchestra']
    for idx in range(1, self.artists_name_split.shape[1]):
      his_values = self.sorted_df[self.sorted_df[f'Artist{idx}'].str.contains('his ', na=False)][f'Artist{idx}'].tolist()
      his_list.extend(his_values)
    his_list.remove('memphis bleek')
    his_list = list(set(his_list))
    for col in self.sorted_df.columns:
      self.sorted_df[col] = self.sorted_df[col].apply(lambda x: None if x in his_list else x)
      
  def generate_collaborations(self):
    artist_columns = [col for col in self.sorted_df.columns if col.startswith('Artist')]
    reshaped_df = self.sorted_df.melt(value_vars=artist_columns, value_name='Artist')
    all_artists = reshaped_df['Artist'].dropna().unique().tolist()

    for artist in tqdm(all_artists, desc="Generating collaborations"):
      collaborators = []
      for idx in range(self.artists_name_split.shape[1]):
        collaborator_rows = self.sorted_df[self.sorted_df[f'Artist{idx}'] == artist]
        if not collaborator_rows.empty:
          collabs = collaborator_rows.iloc[:, 3:].values.flatten().tolist()
          collaborators.extend(collabs)
      collaborators = list(set(collaborators) - {None, artist})
      self.artist_collaboration_dict[artist] = collaborators


  def process(self):
    print("Loading and processing data...")
    self.load_and_process_data()
    self.remove_his_list()
    print("Generating collaborations...")
    self.generate_collaborations()
    self.update_collaborations()
    print("Grouping songs per artist...")
    self.generate_songs_per_artist()
    print("Splitting datasets...")
    self.split_datasets()
    print("Processing complete.")
    


# TODO - Implement this class
class ArtistSplitKpop(ArtistSplit):
  def __init__(self, df_path, out_path, seed=77):
    super().__init__(df_path, out_path, seed)



if __name__ == '__main__':
  processor = ArtistSplitBillboard('csv/billboard_hot100_chosen.csv', 'csv/billboard_artist_split.json')
  processor.process()
  processor.save_to_json(processor.out_path)
  print("Data has been processed and saved to 'csv/billboard_split.json'!")