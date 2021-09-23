"""
Process photos and lineart into obverse and reverse.

Store data

"""
import argparse
import os

import cv2
import numpy as np

from lineart_generator.data_munging.cdli_data_download import RAW_DATA_DIR


PROCESSED_DATA_DIR = os.path.join(os.path.dirname(RAW_DATA_DIR),
                                  'processed_data')


def raw_data_locations(cdli_id):
    paths = {}
    paths['lineart'] = os.path.join(RAW_DATA_DIR, 'lineart', 'l',
                                    f'{cdli_id}_l.jpg')
    paths['photo'] = os.path.join(RAW_DATA_DIR, 'photo', f'{cdli_id}.jpg')
    return paths


def verified_paths(paths):
    for path in paths:
        if not os.path.exists(path):
            return False
    return True


def lineart_labelled_components(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, im_thresh = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    im_dilate = cv2.dilate(im_thresh, kernel, iterations=1)

    # floodfill
    im_floodfill = im_dilate.copy()
    h, w = im_dilate.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    im_out = im_dilate | im_floodfill_inv

    connectivity = 8
    num_comp, labels, stats, centroids = cv2.connectedComponentsWithStats(
                                                im_out,
                                                connectivity,
                                                cv2.CV_32S)
    return num_comp, labels, stats, centroids


def fatcross_labelled_components(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, im_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    im_erode = cv2.erode(im_thresh, kernel, iterations=5)

    connectivity = 8
    num_comp, labels, stats, centroids = cv2.connectedComponentsWithStats(
                                                im_erode,
                                                connectivity,
                                                cv2.CV_32S
                                                )
    return num_comp, labels, stats, centroids


def fatcross_faces(stats):
    areas = []
    for obj in range(stats.shape[0]):
        if obj >= 1:
            areas.append((obj, stats[obj, 4]))
    sorted_areas = sorted(areas, key=lambda x: x[1])
    face_labels = [x[0] for x in sorted_areas[-2:]]
    obv, rev = min(face_labels), max(face_labels)
    return obv, rev


def lineart_faces(stats):
    obv, rev = 1, 2
    return obv, rev


def crop_face(img, face_stats):
    im_height, im_width = img.shape[:2]
    xmin, ymin, width, height, _ = face_stats
    crop_min_x, crop_max_x = max(0, xmin), min(im_width, xmin + width)
    crop_min_y, crop_max_y = max(0, ymin), min(im_height, ymin + height)
    crop_img = img[crop_min_y:crop_max_y, crop_min_x:crop_max_x]
    return crop_img


def process_faces(cdli_id, data_set_name):
    paths = raw_data_locations(cdli_id)

    if verified_paths(list(paths.values())):
        lineart_img = cv2.imread(paths['lineart'])
        _, l_labels, l_stats, l_centroids = lineart_labelled_components(lineart_img)
        l_obv_index, l_rev_index = lineart_faces(l_stats)

        photo_img = cv2.imread(paths['photo'])
        _, f_labels, f_stats, f_centroids = fatcross_labelled_components(photo_img)
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
        help='Parent directory to which processed images will be saved. If "--name" is also entered, images will be saved to <output_dir>\\<name>.',
        )

    parser.add_argument(
        '--name',
        default='',
        help='Name of processed data set. If present, all processed data will be saved into a parent directory with this name.',
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
