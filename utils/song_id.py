import re
import pandas as pd

def get_year_from_date(date):
    if isinstance(date, str):
        return int(date.split('/')[-1])
    elif isinstance(date, pd.Series):
        return date.apply(lambda x: int(x.split('/')[-1]))
    else:
        raise ValueError(f"date should be either str or pd.Series, but got {type(date)}")


def get_era_from_year(year):
    year = max(year, 1960)
    year = min(year, 2010)
    era = (year - 1960) // 10
    return era


def get_song_id(date, song, artist):
    year = get_year_from_date(date)
    song = song.replace('/', ' ')
    artist = artist.replace('/', ' ')
    return f'{{{year}}}_{{{song}}}_{{{artist}}}'


def get_era_from_song_id(song_id): # TODO: remove this
    pattern = r"\{([^}]*)\}"
    matches = re.findall(pattern, song_id)
    year = int(matches[0])
    era = get_era_from_year(year)
    return era


def decode_song_id(song_id):
    pattern = r"\{([^}]*)\}"
    matches = re.findall(pattern, song_id)
    assert len(matches) == 3, f"song_id should have 3 parts, but got {len(matches)} for {song_id}"
    return matches # year, song, artist