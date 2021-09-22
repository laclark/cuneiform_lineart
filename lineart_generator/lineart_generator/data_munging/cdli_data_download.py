import os

import pandas as pd
import requests


HOST = 'https://cdli.ucla.edu/dl/'

# Other lineart types exist, e.g. ld, ls, le
# These can be added to the im_types below following the same pattern
IM_TYPES = {
    'photo': 
        {
            'type': 'photo',
            'sub-type': '',
            'format': 'jpg',
            }, 
    'lineart_l':
        {
            'type': 'lineart',
            'sub-type': 'l',
            'format': 'jpg',
            }, 
    }


curr_path = os.path.abspath(__file__)
path_sections = curr_path.split(os.path.sep)
DATA_DIR = os.path.join(os.path.sep.join(path_sections[:-4]), 'data', 'raw_data_TEST')


class HaltException(Exception): 
    pass


def cdli_catalogue_paths(cdli_dir):
    if os.path.exists(cdli_dir):
        return [os.path.join(cdli_dir, file) for file in os.listdir(cdli_dir) if file.lower().endswith('.csv') and file.lower().startswith('cdli_catalogue')]
    else: 
        raise RuntimeError('CDLI data directory does not exist.')


def catalogue_concatenation(catalogue_paths):
    """ Note that we are relying on good CDLI data (CSVs readable, aligned CSV cols)
    """
    # only 1st CSV includes header rows
    catalogue_paths.sort() # stringsort will work until 10 catalogue parts

    cdli_catalogue_num = int(catalogue_paths[0][-5])
    assert cdli_catalogue_num == len(catalogue_paths), 'Number of catalogues detected does not match CDLI reported value.'

    dfs = []
    for i, path in enumerate(catalogue_paths):
        if i == 0:
            df = pd.read_csv(path, dtype=str)
            cols = df.columns
        else: 
            df = pd.read_csv(path, header=None, dtype=str)
            assert len(df.columns) == len(cols), 'Number of columns does not match across sub-catalogues.'
        dfs.append(df)
    
    if len(dfs) == 0:
        raise RuntimeError('No CDLI data catalogues available.')
    elif len(dfs) == 1:
        return dfs[0]
    else:
        catalogue = pd.concat(dfs, axis=0, ignore_index=True)
        return catalogue
    

def clean_data(df):
    # clean data - remove any duplicate rows
    df = df.fillna('')
    return df


def filter_data(df, filters):
    # will look for any values that contain the given text
    
    for column, allowed in filters.items():
        allowed_entries = []

        for entry in allowed:
            allowed_entries.extend([val for val in df[column].str.lower().unique() if entry.lower() in val])

        df = df[df[column].str.lower().isin(allowed_entries)]
    return df


def agree_download_start(num_records):
    user_choice = input(f'{num_records} tablets found with the selected filter'
                        ' options.\nEnter "y" or "yes" to download: ')

    if user_choice.lower().strip() not in ['y', 'yes']:
        raise HaltException('Image download halted by user. '
                            'Process exiting.')


def make_url(type, sub_type, format, cdli_num):
    if sub_type:
        sub_type = '_' + sub_type
    return f'{HOST}{type}/{cdli_num}{sub_type}.{format}'


def make_save_path(save_dir, type, sub_type, format, cdli_num):
    return os.path.join(save_dir, type, sub_type, f'{cdli_num}.{format}')


def make_data_dirs():
    for val in IM_TYPES.values():
        dir_path = os.path.join(DATA_DIR, val['type'], val['sub-type'])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def download_image(url, save_path):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)


def download_data_set(records, save_dir):
    for i, record in enumerate(records.itertuples()):
        cdli_num = f'P{record.id_text.zfill(6)}'
        print(f'Downloading {cdli_num}, {i + 1} of {len(records)} records')

        for im_type in IM_TYPES:
            type = IM_TYPES[im_type]['type']
            sub_type = IM_TYPES[im_type]['sub-type']
            format = IM_TYPES[im_type]['format']

            url = make_url(type, sub_type, format, cdli_num)
            save_path = make_save_path(save_dir, type, sub_type, format, cdli_num)

            download_image(url, save_path)


def download_selected_tablet_images(cdli_dir, filters, max_ims):
    paths = cdli_catalogue_paths(cdli_dir)

    catalogue = catalogue_concatenation(paths)
    cleaned_catalogue = clean_data(catalogue)

    records = filter_data(cleaned_catalogue, filters)
    max_ims = min(len(records), max_ims)

    agree_download_start(max_ims) 
    records = records.iloc[:max_ims]

    make_data_dirs()

    download_data_set(records, DATA_DIR)


if __name__ == '__main__':

    cdli_dir = 'D:\\Projects\\cdli_cuneiform\\data'
    filters = {
        'collection': ['hearst'],
        'language': ['Sumerian'],
        'surface_preservation': ['good'],
        }

    max_ims = 5
            
    download_selected_tablet_images(cdli_dir, filters, max_ims)