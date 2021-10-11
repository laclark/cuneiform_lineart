"""Downloads raw image data (photos and line art) for selected CDLI records.

Tablets are selected for download in one of two ways:
    1. Given the CDLI catalogue directory and filters, download
            artifacts for selected records up to the maximum allowed number.
    2. Read a text document listing desired CDLI numbers, and download
            artifacts for each tablet on the list.

If both cdli_dir and cdli_list_path are given, the text document will be
given priority.

In the first usage, the script will first combine all catalogue entries
(e.g. 'cdli_catalogue_1of2.csv' and 'cdli_catalogue_2of2.csv').  Do not
rename the catalogue files. Optional filters supplied by the user via command
line arguments ('collection', 'language', or 'preservation') are then applied
to the catalogue, and the first N (optional user input) matching records are
selected for download.

In both usages, the script will attempt to download each selected tablet's
associated photographic record and line art.  Note that for a given entry,
EITHER photo OR lineart, BOTH, or NEITHER image(s) may be downloaded
(depending on available data).

Downloaded images will be stored as follows:

    cuneiform
    ├── data
    │   └── raw_data
    │   │   ├── lineart
    │   │   │   └── l
    │   │   │   │   └── <cdli_num>_l.jpg
    │   │   └── photo
    │   │   │   └── <cdli_num>.jpg


Examples:

    * Download data for first 10 tablets from CDLI catalogue.

        python cdli_data_download.py            \
            --cdli_dir=<path_to_CDLI_data>      \
            --max_tablets=10

    * Download five tablets from the collections at the Hearst Museum of
        Anthropology (Berkeley, CA, USA), in the Sumerian language,
        with good surface preservation.

        python cdli_data_download.py            \
            --cdli_dir=<path_to_CDLI_data>      \
            --collection hearst                 \
            --language sumerian                 \
            --preservation good                 \
            --max_tablets=5                     \

    * Download data for tablets listed in text document.

        python cdli_data_download.py            \
            --cdli_list_path=<path_to_text_list>

Todo:
    * Check whether image exists prior to download; don't download again.

"""

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
RAW_DATA_DIR = os.path.join(os.path.sep.join(path_sections[:-4]),
                                             'data',
                                             'raw_data')


class HaltException(Exception):
    pass


def combine_filters(collections, languages, preservations):
    """Make filter dictionary from user inputs.  Used to filter catalogue data.

    Args:
        collections (list of str): List of terms that should be ....
        languages (list of str):
        preservations (list of str):

    Return:
        filters (dict): Dictionary with keys equal to catalogue column names,
            values equal to list of strings that should be contained.
    """
    filters = {}

    if len(collections) > 0:
        filters['collection'] = collections

    if len(languages) > 0:
        filters['language'] = languages

    if len(preservations) > 0:
        filters['surface_preservation'] = preservations

    return filters


def cdli_catalogue_paths(cdli_dir):
    """Find catalogue paths within cdli data directory."""
    if os.path.exists(cdli_dir):
        return [os.path.join(cdli_dir, file) for file in os.listdir(cdli_dir)
                if file.lower().endswith('.csv')
                and file.lower().startswith('cdli_catalogue')]
    else:
        raise RuntimeError('CDLI data directory does not exist.')


def catalogue_concatenation(catalogue_paths):
    """Concatenate catalogue subsections into single dataframe.

    Note that we are relying on the correctness of the CDLI data, e.g. all CSVs
    must contain the same number of columns in the same order.  This is because
    column headers are only included in the first subsection.

    """
    # only 1st CSV includes header rows
    catalogue_paths.sort()  # string sort will work until 10 catalogue sections

    cdli_catalogue_num = int(catalogue_paths[0][-5])

    assert cdli_catalogue_num == len(catalogue_paths), 'Number of catalogues detected does not match CDLI reported value.'

    subsections = []
    for i, path in enumerate(catalogue_paths):
        if i == 0:
            subsection = pd.read_csv(path, dtype=str)
            cols = subsection.columns
        else:
            subsection = pd.read_csv(path, header=None, dtype=str)
            assert len(subsection.columns) == len(cols), 'Number of columns does not match across sub-catalogues.'
        subsections.append(subsection)

    if len(subsections) == 0:
        raise RuntimeError('No CDLI data catalogues available.')
    elif len(subsections) == 1:
        return subsections[0]
    else:
        catalogue = pd.concat(subsections, axis=0, ignore_index=True)
        return catalogue


def clean_data(records):
    """Minimal cleaning - convert NANs to empty strings."""
    records = records.fillna('')
    return records


def filter_data(records, filters):
    """ Returns records with column entries that contain user-provided terms.

    For each key (column name) in 'filters', a list of required terms is given.
    Returned records must contain at least one of the listed terms per
    key/column.  For any catalogue column that is not a key in the dictionary,
    no filtering will occur.  An empty filter dictionary returns all records.

    Args:
        records (pd.DataFrame): Dataframe containing CDLI records.
        filters (dict of str -> list): Maps catalogue column names to a list of
            required terms.

    Returns:
        records (pd.DataFrame): Dataframe containing CDLI records after filtering.

    """
    for column, allowed in filters.items():
        allowed_entries = []

        for entry in allowed:
            allowed_entries.extend([val for val
                                    in records[column].str.lower().unique()
                                    if entry.lower() in val])

        records = records[records[column].str.lower().isin(allowed_entries)]
    return records


def agree_download_start(num_records, max_download):
    """Require user input to start data download."""
    user_choice = input(f'{num_records} tablets found with the selected filter'
                        f' options.\nData for {max_download} tablets will be '
                        'downloaded.\nEnter "y" or "yes" to download: ')

    if user_choice.lower().strip() not in ['y', 'yes']:
        raise HaltException('Image download halted by user. '
                            'Process exiting.')


def make_url(type, sub_type, format, cdli_num):
    """CDLI urls for image download."""
    if sub_type:
        sub_type = '_' + sub_type
    return f'{HOST}{type}/{cdli_num}{sub_type}.{format}'


def make_save_path(save_dir, type, sub_type, format, cdli_num):
    """Path for local image file."""
    if sub_type:
        file_suffix = '_' + sub_type
    else:
        file_suffix = sub_type
    return os.path.join(save_dir, type, sub_type,
                        f'{cdli_num}{file_suffix}.{format}')


def make_raw_data_dirs():
    """Create directory structure for raw data types."""
    for val in IM_TYPES.values():
        dir_path = os.path.join(RAW_DATA_DIR, val['type'], val['sub-type'])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def download_image(url, save_path):
    """Download tablet image (photo or line art).

    Todo:
        * Add retries with timeout.
    """
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


def download_data_set(cdli_nums, save_dir):
    """Download photo and line art for given tablets.

    Locally stored files are separated into subdirectories based on image type
    ('photo', 'lineart_l').

    Args:
        cdli_nums (list of str): List of CDLI tablet numbers.
        save_dir (str): Path to parent directory for local image storage.

    Returns:
        downloaded_data (dict of str -> list of str): Maps image type to list
            of CDLI numbers designating successfully downloaded images.

    """
    downloaded_data = {}

    for i, cdli_num in enumerate(cdli_nums):
        print(f'Downloading {cdli_num}, {i + 1} of {len(cdli_nums)} records')

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


def read_tablet_text(path):
    """Todo: Add cdli number format verification."""
    with open(path, 'r') as f:
        cdli_nums = []
        for line in f.readlines():
            cdli_nums.append(line.strip('\n'))
    return cdli_nums


def select_tablets_from_catalogue(cdli_dir, filters, max_ims):
    """Given the CDLI data directory, filters, and maximum allowable number of
    tablets, return selected CDLI numbers.

    """
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

    cdli_nums = [f'P{val}' for val in
                 records['id_text'].str.zfill(6).to_list()]
    return cdli_nums


def download_selected_tablet_images(cdli_dir, cdli_list_path, filters,
                                    max_ims=None):
    """Downloads tablet artifacts (photographs and line art) in one of two
    ways:
        1. Reads CDLI catalogue from directory, applies filters, and downloads
           artifacts for selected records up to the maximum allowed number.
        2. Reads a text document of CDLI numbers and downloads artifacts for
           each tablet on the list.

    If both cdli_dir and cdli_list_path are given, the text document will be
    given priority.

    Args:
        cdli_dir (str): Directory containing CDLI data catalogues.
        cdli_list_path (str): Path to list of CDLI numbers for selected
            tablets.  
        filters (dict of str -> list): Maps catalogue column names to a list of
            required terms.
        max_ims (int or None): Maximum number of records to download images
            for; if None, will download images for all filtered records.

    Returns:
        local_ids: Maps image type to list
            of CDLI numbers designating successfully downloaded images.

    """
    if cdli_list_path is not None:
        cdli_nums = read_tablet_text(cdli_list_path)
    elif cdli_dir is not None:
        cdli_nums = select_tablets_from_catalogue(cdli_dir, filters, max_ims)
    else:
        raise RuntimeError(('Cannot download data if both "cdli_dir" and'
                            ' "cdli_list_path" are undefined.  Exiting.'))

    make_raw_data_dirs()

    local_ids = download_data_set(cdli_nums, RAW_DATA_DIR)

    return local_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cdli_dir',
        help='Root directory of CDLI data repository.',
        )

    parser.add_argument(
        '--cdli_list_path',
        help=('Path to text document containing list of tablets (cdli numbers)'
              'for which images should be downloaded.'),
        )

    parser.add_argument(
        '--collection',
        nargs='+',
        default=[],
        help='Filter available tablets by collection(s).',
        )

    parser.add_argument(
        '--language',
        nargs='+',
        default=[],
        help='Filter available tablets by language(s).',
        )

    parser.add_argument(
        "--preservation",
        nargs='+',
        default=[],
        help='Filter available tablets by preservation label(s).',
        )

    parser.add_argument(
        "--max_tablets",
        help=('Maximum number of tablets for which images will be downloaded'
              ' (first N in selected records).'),
        )

    args = parser.parse_args()

    cdli_dir = args.cdli_dir
    cdli_list_path = args.cdli_list_path
    collections = args.collection
    languages = args.language
    preservations = args.preservation
    max_ims = args.max_tablets

    if max_ims is not None:
        max_ims = int(max_ims)

    filters = combine_filters(collections, languages, preservations)

    download_selected_tablet_images(cdli_dir, cdli_list_path, filters, max_ims)
