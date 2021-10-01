"""Processes photographs and line art to create matched faces for training.

Given a directory of raw CDLI data, the script matches photographs and
line art for available tablets based on image name (any tablets without both
types of image will be ignored).  For each match, morphological operations are
used to subdivide the images into single faces, and obverse and reverse faces
are identified.

Note that the morphological operations and image dissection/matching rules used
in this script have been developed for:
    * full CDLI-style photographic fatcrosses
    * line art comprising cuneiform signs and tablet outlines (complete or
        nearly complete) for the obverse (top drawing) and reverse (bottom
        drawing) tablet faces.

Other styles of lineart or alternative photographic layouts are likely to be
incorrectly processed.

See README figures for more detailed explanations and
https://cdli.ox.ac.uk/wiki/doku.php?id=submission_guidelines for information
about CDLI image creation.

Processed images are stored as follows:

    cuneiform
    ├── data
    │   └── processed_data
    │   │   ├── <data_set_name - optional>
    │   │   │   ├── lineart_obv
    │   │   │   │   └── <cdli_num>.jpg
    │   │   │   ├── lineart_rev
    │   │   │   │   └── <cdli_num>.jpg
    │   │   │   ├── photo_obv
    │   │   │   │   └── <cdli_num>.jpg
    │   │   │   └── photo_rev
    │   │   │   │   └── <cdli_num>.jpg


Example:

    * Process all data in default raw_data directory, and output into
    default processed data directory with given name.

        python cdli_data_download.py    \
            --name=<data_set_name>      \

"""

import argparse
import os

import cv2
import numpy as np

from lineart_generator.data_munging.cdli_data_download import RAW_DATA_DIR


PROCESSED_DATA_DIR = os.path.join(os.path.dirname(RAW_DATA_DIR),
                                  'processed_data')


def raw_data_locations(cdli_id):
    """Make paths to raw data storage locations."""
    paths = {}
    paths['lineart'] = os.path.join(RAW_DATA_DIR, 'lineart', 'l',
                                    f'{cdli_id}_l.jpg')
    paths['photo'] = os.path.join(RAW_DATA_DIR, 'photo', f'{cdli_id}.jpg')
    return paths


def verified_paths(paths):
    """Check all paths exist; in this case line art and photo."""
    for path in paths:
        if not os.path.exists(path):
            return False
    return True


def lineart_components(img):
    """ Segment line art image into tablet faces and return shape information.

    Args:
        img (np.ndarray, dtype 'uint8', dimensions m x n x 3): Line art image.

    Returns:
        num_comp (int): Number of labelled components, including background.
        labels (np.ndarray, dtype 'int32', dimensions m x n): Image of
            connected components with pixel values equal to component labels.
            Compare to line art image to check success of segmentation.
        stats (np.ndarray, dtype 'int32', num_comp x 5): Component statistics,
            with row index equal to label and columns in order as follows:
                * leftmost (x) coordinate of component
                * topmost (y) coordinate of component
                * width (pixels) of bounding box
                * height (pixels) of bounding box
                * total area (pixels) of component
        centroids (np.ndarray, dtype 'float64',  num_comp x 2): X, Y coordinate
            pairs for component centroids (row index is label).

    """
    # Convert line art to gray-scale, m x n
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold (Otsu) inverted image
    _, im_thresh = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Dilation to fill small gaps in line art (signs or outline)
    kernel = np.ones((5, 5), np.uint8)
    im_dilate = cv2.dilate(im_thresh, kernel, iterations=1)

    # Floodfill to segment faces
    im_floodfill = im_dilate.copy()
    h, w = im_dilate.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)   # Fills img outside faces
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)   # Fills faces, not signs

    # Combine dilated, filled to select both writing and white space in faces
    im_out = im_dilate | im_floodfill_inv

    # Get connected component information
    connectivity = 8   # 8-connectivity for diagonally touching pixels
    num_comp, labels, stats, centroids = cv2.connectedComponentsWithStats(
                                                im_out,
                                                connectivity,
                                                cv2.CV_32S)
    return num_comp, labels, stats, centroids


def fatcross_components(img):
    """Segment fatcross image into tablet faces and return shape information.

    Args:
        img (np.ndarray, dtype 'uint8', dimensions m x n x 3): Line art image.

    Returns:
        num_comp (int): Number of labelled components, including background.
        labels (np.ndarray, dtype 'int32', dimensions m x n): Image of
            connected components with pixel values equal to component labels.
            Compare to line art image to check success of segmentation.
        stats (np.ndarray, dtype 'int32', num_comp x 5): Component statistics,
            with row index equal to label and columns in order as follows:
                * leftmost (x) coordinate of component
                * topmost (y) coordinate of component
                * width (pixels) of bounding box
                * height (pixels) of bounding box
                * total area (pixels) of component
        centroids (np.ndarray, dtype 'float64',  num_comp x 2): X, Y coordinate
            pairs for component centroids (row index is label).

    """
    # Convert to grayscale + threshold, no inversion as background is black
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, im_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Erosion to remove noise
    kernel = np.ones((5, 5), np.uint8)
    im_erode = cv2.erode(im_thresh, kernel, iterations=5)

    # Get connected component information
    connectivity = 8
    num_comp, labels, stats, centroids = cv2.connectedComponentsWithStats(
                                                im_erode,
                                                connectivity,
                                                cv2.CV_32S
                                                )
    return num_comp, labels, stats, centroids


def fatcross_faces(stats):
    """Use labelled component statistics to select obverse/reverse faces.

    Ignoring the background (first labelled component), select the two largest
    components by pixel area.  These should correspond to the obverse and
    reverse faces for a 'standard' tablet shape (assuming good segmentation).
    Obverse and reverse labels are assigned by sorting label values, as the
    obverse face is above the reverse and should have a lower label.

    If the tablet is not photographed on a black background, has a particularly
    unusual shape, or is too fragmented, incorrect results are likely.

    Args:
        stats (np.ndarray, dtype 'int32', num_comp x 5): Component statistics,
            as described in fatcross_components.

    Returns:
        obv (int): Label value corresponding to obverse face.
        rev (int): Label value corresponding to reverse face.

    """
    areas = []
    for obj in range(stats.shape[0]):
        if obj >= 1:   # ignore background
            areas.append((obj, stats[obj, 4]))
    sorted_areas = sorted(areas, key=lambda x: x[1])
    face_labels = [x[0] for x in sorted_areas[-2:]]
    obv, rev = min(face_labels), max(face_labels)
    return obv, rev


def lineart_faces(stats):
    """Assign line art faces assuming good segmentation.

    Caveats about style and shape are as described in module and fatcross_faces
    docstrings.

    """
    obv, rev = 1, 2
    return obv, rev


def crop_face(img, face_stats):
    """Crop selected area (tablet face) from full input image.
    Args:
        img (np.ndarray, dtype 'uint8', dimensions m x n x 3): Full photo/line
            art image.
        face_stats (np.ndarray, dtype 'int32', dimensions num_comp): Component
            statistics for a single face (see fatcross_components for
            stat list).

    Returns
        crop_img (np.ndarray, dtype 'uint8', dimensions i x j x 3): Cropped
            image containing component described by face_stats.

    """
    im_height, im_width = img.shape[:2]
    xmin, ymin, width, height, _ = face_stats
    crop_min_x, crop_max_x = max(0, xmin), min(im_width, xmin + width)
    crop_min_y, crop_max_y = max(0, ymin), min(im_height, ymin + height)
    crop_img = img[crop_min_y:crop_max_y, crop_min_x:crop_max_x]
    return crop_img


def process_faces(cdli_id, data_set_name):
    """Select and save obverse + reverse faces from photo + line art for a
    single tablet.

    Tablet is only processed if both photo and line art are saved locally.

    Args:
        cdli_id (str): CDLI number, e.g. 'P000001'.
        data_set_name (str): Name of data set.  Images will be stored in
            subdirectory of this name within the parent processed data
            directory.
    Returns:
        None

    """

    paths = raw_data_locations(cdli_id)

    if verified_paths(list(paths.values())):
        lineart_img = cv2.imread(paths['lineart'])
        _, l_labels, l_stats, l_centroids = lineart_components(lineart_img)
        l_obv_index, l_rev_index = lineart_faces(l_stats)

        photo_img = cv2.imread(paths['photo'])
        _, f_labels, f_stats, f_centroids = fatcross_components(photo_img)
        f_obv_index, f_rev_index = fatcross_faces(f_stats)

        cropped_line_obv = crop_face(lineart_img, l_stats[l_obv_index])
        cropped_line_rev = crop_face(lineart_img, l_stats[l_rev_index])

        cropped_photo_obv = crop_face(photo_img, f_stats[f_obv_index])
        cropped_photo_rev = crop_face(photo_img, f_stats[f_rev_index])

        # resize lineart faces
        face_width = f_stats[f_obv_index][2]
        face_height = f_stats[f_obv_index][3]

        scale_line_obv = cv2.resize(cropped_line_obv, (face_width, face_height), interpolation=cv2.INTER_CUBIC)
        scale_line_rev = cv2.resize(cropped_line_rev, (face_width, face_height), interpolation=cv2.INTER_CUBIC)

        dirs = {
            'l_obv': os.path.join(PROCESSED_DATA_DIR, data_set_name, 'lineart_obv'),
            'l_rev': os.path.join(PROCESSED_DATA_DIR, data_set_name, 'lineart_rev'),
            'p_obv': os.path.join(PROCESSED_DATA_DIR, data_set_name, 'photo_obv'),
            'p_rev': os.path.join(PROCESSED_DATA_DIR, data_set_name, 'photo_rev'),
        }

        for dir in dirs.values():
            if not os.path.exists(dir):
                os.makedirs(dir)

        cv2.imwrite(os.path.join(dirs['l_obv'], f'{cdli_id}.jpg'), scale_line_obv)
        cv2.imwrite(os.path.join(dirs['l_rev'], f'{cdli_id}.jpg'), scale_line_rev)
        cv2.imwrite(os.path.join(dirs['p_obv'], f'{cdli_id}.jpg'), cropped_photo_obv)
        cv2.imwrite(os.path.join(dirs['p_rev'], f'{cdli_id}.jpg'), cropped_photo_rev)


def process_tablets(cdli_ids, data_set_name):
    """Attempt face extraction and matching for each tablet."""
    for i, id in enumerate(cdli_ids):
        print(f'Processing {id}, {i + 1} of {len(cdli_ids)} tablets')
        process_faces(id, data_set_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--raw_data_dir',
        help='Root directory containing downloaded CDLI images.',
        )

    parser.add_argument(
        '--output_dir',
        help=('Parent directory to which processed images will be saved. If'
              ' "--name" is also entered, images will be saved to <output_dir>'
              '\\<name>.'),
        )

    parser.add_argument(
        '--name',
        default='',
        help=('Name of processed data set. If present, all processed data will'
              ' be saved into a parent directory with this name.'),
        )

    args = parser.parse_args()

    raw_data_dir = args.raw_data_dir
    output_dir = args.output_dir
    data_set_name = args.name

    if raw_data_dir is not None:
        RAW_DATA_DIR = raw_data_dir

    if output_dir is not None:
        PROCESSED_DATA_DIR = output_dir

    photo_dir = os.path.join(RAW_DATA_DIR, 'photo')
    cdli_ids = [name[:-4] for name in os.listdir(photo_dir)]

    process_tablets(cdli_ids, data_set_name)
