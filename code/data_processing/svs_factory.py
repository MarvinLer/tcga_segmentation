__author__ = 'marvinler'

import os
import subprocess
import numpy as np
from tqdm import tqdm
import imageio
import shutil
from concurrent import futures

from code import N_PROCESSES
from ext import deepzoom_tile


def download_svs_files(gdc_executable_filepath, manifest_filepath, output_dir):
    # Opens manifest and retrieves the ids to be retrieved
    with open(manifest_filepath, 'r') as f:
        lines = f.read().splitlines()[1:]
    cells = [line.split('\t') for line in lines]
    ids = [cell[0] for cell in cells]
    filenames = [cell[1] for cell in cells]
    md5sums = [cell[2] for cell in cells]

    # Extract patient id based on SVS names
    cases_ids = ['-'.join(filename.split('-')[:3]) for filename in filenames]

    for file_id in tqdm(ids):
        # Download file if not already downloaded
        if not os.path.exists(os.path.join(output_dir, file_id)):
            subprocess.check_output([gdc_executable_filepath, 'download', '--dir', output_dir,
                                     '--n-processes', str(N_PROCESSES), file_id])

    # Infer resulting SVS filepaths from output_dir, id and filename of files
    resulting_svs_filepaths = [os.path.abspath(os.path.join(output_dir, id_file, filename))
                               for id_file, filename in zip(ids, filenames)]

    assert len(resulting_svs_filepaths) == len(ids) == len(md5sums) == len(cases_ids)

    return resulting_svs_filepaths, md5sums, cases_ids


def list_slides_in_folder(input_folder, with_supfolder=False):
    slides_format = ('.svs', '.vms', '.vmu', '.ndpi', '.scn', '.mrxs', '.tif', '.tiff', '.bif')

    all_files_folders = list(map(lambda f: os.path.join(input_folder, f), os.listdir(input_folder)))
    if with_supfolder:
        only_folders = list(filter(lambda f: not os.path.isfile(f), all_files_folders))
        # list all slides within each folder of the input folder
        all_slides = [list(map(lambda f: os.path.join(folder, f),
                               list(filter(lambda f: f.endswith(slides_format), os.listdir(folder)))))
                      for folder in only_folders]
        # flatten
        all_slides = [slide for slide_group in all_slides for slide in slide_group]
        return all_slides

    only_files = list(filter(os.path.isfile, all_files_folders))
    return list(filter(lambda f: f.endswith(slides_format), only_files))


def tile_slide(slide_filepath, desired_tile_with, desired_overlap, desired_magnification):
    output_folder = os.path.join(os.path.dirname(slide_filepath),
                                 '.'.join(os.path.basename(slide_filepath).split('.')[:-1]))

    # Call external script that computes full-scales extraction of tiles + static viewer
    dz_tiler = deepzoom_tile.DeepZoomStaticTiler(slide_filepath, output_folder, 'png',
                                                 desired_tile_with, desired_overlap, True, 90,
                                                 N_PROCESSES, True)
    if not os.path.exists(output_folder):
        dz_tiler.run()

    # Find magnification of slide
    slide_properties = dz_tiler._slide.properties
    if 'openslide.objective-power' not in slide_properties._keys():
        raise ValueError('Did not find slide property \'openslide.objective-power\'')
    slide_magnification = float(slide_properties['openslide.objective-power'])
    if desired_magnification > slide_magnification:
        raise ValueError('Slide magnification of %s is above desired magnification of %d' % (str(slide_magnification),
                                                                                             desired_magnification))
    # Compute minimal power of 2 dividing slide magnification into desired magnification
    best_power_of_2 = 0
    while True:
        if desired_magnification * (2 ** best_power_of_2) >= slide_magnification:
            break
        best_power_of_2 += 1
    # Return the corresponding DeepZoom zoom folder
    dz_level_folders = list(filter(lambda f: not os.path.isfile(os.path.join(output_folder, 'slide_files', f)),
                                   os.listdir(os.path.join(output_folder, 'slide_files'))))
    best_corresponding_level_folder = sorted(list(map(int, dz_level_folders)), reverse=True)[best_power_of_2]

    return os.path.join(output_folder, 'slide_files', str(best_corresponding_level_folder))


def tile_slides(slides_filepaths, desired_tile_with, desired_overlap, desired_magnification):
    """ Wrapper for tile_slide() for multiple slides """
    containing_folders = []
    for slide_filepath in slides_filepaths:
        containing_folders.append(tile_slide(slide_filepath, desired_tile_with, desired_overlap, desired_magnification))
    return containing_folders


def _compute_pixelwise_is_background(jpeg_file, background_pixel_value):
    """ Load image, then return pixel-wise True or False if all RGB values are above provided value """
    img = imageio.imread(jpeg_file)

    channel_above_threshold = img > background_pixel_value
    pixel_above_threshold = np.prod(channel_above_threshold, axis=-1)

    return img, pixel_above_threshold


def is_tile_mostly_background(img_filepath, background_pixel_value, background_threshold, expected_shape):
    """ Returns True if tile percent of background pixels are above background_threshold or tile is not of shape
    expected_shape.

    :param img_filepath: abs path to jpeg/jpg/png image
    :param background_pixel_value: threshold above which a channel pixel is considered background
    :param background_threshold: percent above which a tile is considered background based on is pixel background
    :param expected_shape: expected shape of tile
    """
    img, pixel_above_threshold = _compute_pixelwise_is_background(img_filepath, background_pixel_value)
    if img.shape != expected_shape:
        return True, 0.

    percent_background_pixels = np.sum(pixel_above_threshold) / (img.shape[0] * img.shape[1])
    return percent_background_pixels > background_threshold, percent_background_pixels


def move_and_filter_tiles_folders(tiles_folders, classes, slides_id, cases_ids, output_folder, background_pixel_value,
                                  background_threshold, expected_shape, logger):
    """
    Move all tiles from input tiles folders into new folders at provided dest folder by discarding tiles considered as
    background, and also by writing a summary file with all original tiles names, and a label file with associated
    class.
    :param tiles_folders: list of input tiles folders, each containing tiles from WSI
    :param classes: list of integer corresponding to WSI classes
    :param slides_id: WSI slides ids
    :param cases_ids: WSI associated cases IDs for further case-wise dataset splitting
    :param output_folder: parent folder in which to create 1 folder per each input WSI
    :param background_pixel_value: threshold above which a channel pixel is considered background
    :param background_threshold: percent above which a tile is considered background based on is pixel background
    :param expected_shape: expected shape of tile; tiles of different shape are discarded (e.g. extrema ones)
    :param logger: logger object to print some info
    :return: list of output processed slides folders, 1 per input WSI except for bugged WSI
    """
    def move_jpeg_file(inputs):
        slide_id, img_filepath = inputs[0], inputs[1]
        is_mostly_background, percent_background = is_tile_mostly_background(img_filepath=img_filepath,
                                                                             background_pixel_value=background_pixel_value,
                                                                             background_threshold=background_threshold,
                                                                             expected_shape=expected_shape)

        # If img considered not mostly background, move to processed folder, otherwise is discarded
        if not is_mostly_background:  # copy to dest folder
            new_filepath = os.path.join(output_folder, slide_id, os.path.basename(img_filepath))
            shutil.copyfile(os.path.abspath(img_filepath),
                            os.path.abspath(new_filepath))

    assert len(tiles_folders) == len(classes) == len(slides_id) == len(cases_ids)

    destination_folders = []
    for i, (tile_folder, slide_id, class_, case_id) in \
            tqdm(enumerate(zip(tiles_folders, slides_id, classes, cases_ids)), total=len(tiles_folders)):
        new_folderpath = os.path.abspath(os.path.join(output_folder, slide_id))
        destination_folders.append(new_folderpath)
        # if destination folder already exists then folder already processed -> skip
        if os.path.exists(new_folderpath):
            continue
        os.makedirs(new_folderpath)

        images_filenames = [f for f in os.listdir(tile_folder) if f.endswith(('.jpeg', '.jpg', '.png', '.pt'))]

        # Write a summary.txt file containing all the tiles of WSI, before background discarding
        with open(os.path.join(new_folderpath, 'summary.txt'), 'w') as f:
            f.write('\n'.join(images_filenames))
        # Write file containing label as int
        with open(os.path.join(new_folderpath, 'label.txt'), 'w') as f:
            f.write(str(class_))
        # Write file containing case id
        with open(os.path.join(new_folderpath, 'case_id.txt'), 'w') as f:
            f.write(case_id)

        # Add slide if within arguments of move_jpeg_file
        try:
            with futures.ThreadPoolExecutor(max_workers=N_PROCESSES) as pool:
                images_filepaths = [(slide_id, os.path.join(tile_folder, img_filename))
                                    for img_filename in images_filenames]
                list(pool.map(move_jpeg_file, images_filepaths))
        except (SyntaxError, ValueError) as e:
            logger.warn('  discarding %s because some image files are corrupted: %s' % (slide_id, e))
            continue

    # return all destination slides folders
    return list(map(os.path.abspath, destination_folders))
