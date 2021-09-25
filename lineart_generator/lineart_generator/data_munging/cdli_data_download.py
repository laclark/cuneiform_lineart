import argparse
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
RAW_DATA_DIR = os.path.join(os.path.sep.join(path_sections[:-4]), 'data', 'raw_data')


class HaltException(Exception): 
    pass


def combine_filters(collections, languages, preservations):
    filters = {}

    if collections is not None:
        filters['collection'] = collections

    if languages is not None:
        filters['language'] = languages

    if preservations is not None > 0:
        filters['surface_preservation'] = preservations    

    return filters


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


def agree_download_start(num_records, max_download):
    user_choice = input(f'{num_records} tablets found with the selected filter'
                        f' options.\nData for {max_download} tablets will be '
                        'downloaded.\nEnter "y" or "yes" to download: ')

    if user_choice.lower().strip() not in ['y', 'yes']:
        raise HaltException('Image download halted by user. '
                            'Process exiting.')


def make_url(type, sub_type, format, cdli_num):
    if sub_type:
        sub_type = '_' + sub_type
    return f'{HOST}{type}/{cdli_num}{sub_type}.{format}'

def make_save_path(save_dir, type, sub_type, format, cdli_num):
    if sub_type:
        file_suffix = '_' + sub_type
    else: 
        file_suffix = sub_type
    return os.path.join(save_dir, type, sub_type, f'{cdli_num}{file_suffix}.{format}')


def make_raw_data_dirs():
    for val in IM_TYPES.values():
        dir_path = os.path.join(RAW_DATA_DIR, val['type'], val['sub-type'])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def download_image(url, save_path):
    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            return True
        else:
            return False
    except:
        return False


def download_data_set(records, save_dir):
    downloaded_data = {}

    for i, record in enumerate(records.itertuples()):
        cdli_num = f'P{record.id_text.zfill(6)}'
        print(f'Downloading {cdli_num}, {i + 1} of {len(records)} records')        

        for im_type in IM_TYPES:
            type = IM_TYPES[im_type]['type']
            sub_type = IM_TYPES[im_type]['sub-type']
            format = IM_TYPES[im_type]['format']

            url = make_url(type, sub_type, format, cdli_num)
            save_path = make_save_path(save_dir, 
                                       type, 
                                       sub_type, 
                                       format, 
                                       cdli_num,
                                       )

            downloaded = download_image(url, save_path)

            if downloaded:
                if im_type not in downloaded_data:
                    downloaded_data[im_type] = []
                downloaded_data[im_type].append(cdli_num)

    return downloaded_data


def download_selected_tablet_images(cdli_dir, filters, max_ims=None):
    paths = cdli_catalogue_paths(cdli_dir)

    catalogue = catalogue_concatenation(paths)
    cleaned_catalogue = clean_data(catalogue)

    records = filter_data(cleaned_catalogue, filters)

    if max_ims is not None:
        max_ims = min(len(records), max_ims)
    else:
        max_ims = len(records)

    agree_download_start(len(records), max_ims)
    records = records.iloc[:max_ims]

    make_raw_data_dirs()

    local_ids = download_data_set(records, RAW_DATA_DIR)

    return local_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cdli_dir',
        help='Root directory of CDLI data repository.',
        required=True,
        )

    parser.add_argument(
        '--collection',
        nargs='+',
        help='Filter available tablets by collection(s).',
        )

    parser.add_argument(
        '--language',
        nargs='+',
        help='Filter available tablets by language(s).',
        )

    parser.add_argument(
        "--preservation",
        nargs='+',
        help='Filter available tablets by preservation label(s).',
        )

    parser.add_argument(
        "--max_tablets",
        help='Maximum number of tablets for which images will be downloaded (first N in selected records).',
        )

    args = parser.parse_args()

    cdli_dir = args.cdli_dir
    collections = args.collection
    languages = args.language
    preservations = args.preservation
    max_ims = int(args.max_tablets)

    filters = combine_filters(collections, languages, preservations)

    download_selected_tablet_images(cdli_dir, filters, max_ims)
