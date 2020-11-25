__author__ = 'marvinler'

import os
import argparse
from time import gmtime, strftime

from code import get_logger
from code.data_processing import case_factory, svs_factory

desired_magnification = 20
desired_tile_width = 224
desired_overlap = 0
expected_tile_shape = (desired_tile_width, desired_tile_width, 3)

background_threshold = .75
background_pixel_value = 220

train_size = .75
val_size = .1
test_size = 1. - train_size - val_size
assert test_size > 0


def get_parser():
    parser = argparse.ArgumentParser(description='Data processing module: handle slides download from The Cancer '
                                                 'Genome Atlas platform, slides tiling, and background removal.')
    parser.add_argument('output_folder', type=str, default=None, metavar='OUTPUT_FOLDER',
                        help='folder containing resulting background-filtered tiles from downloaded or source slides')

    parser.add_argument('--gdc-executable', type=str, default=None, metavar='GDC_EXE_FILEPATH',
                        help='path of the gdc executable used in case of download from TCGA platform')
    parser.add_argument('--manifest', type=str, default=None, metavar='MANIFEST_FILE',
                        help='manifest file downloaded from the TCGA platform containing md5sums and co of the slides '
                             'to be downloaded')

    parser.add_argument('--no-download', action='store_true', default=False,
                        help='specify that no download should be done from TCGA platform; in this case, seeks for any '
                             'slides with openslide-supported format')
    parser.add_argument('--source-slides-folder', type=str, default=None, metavar='GDC_EXE_FILEPATH',
                        help='source folder containing slides to be processed when no TCGA download is performed')
    return parser


def main(output_folder, do_download, gdc_executable, manifest_filepath, source_slides_folder):
    logger = get_logger(filename_handler='code.data_processing.' + os.path.basename(__file__) +
                                         '_' + strftime("%Y-%m-%d_%H:%M:%S", gmtime()) + '.log')

    logger.info('Data_processing control parameters:')
    logger.info('  do_download: ' + str(do_download))
    logger.info('  gdc_executable: ' + str(gdc_executable))
    logger.info('  manifest_filepath: ' + str(manifest_filepath))
    logger.info('  source_slides_folder: ' + str(source_slides_folder))
    logger.info('  Meta-parameters:')
    logger.info('    desired_magnification %s' % str(desired_magnification))
    logger.info('    tile_width %s' % str(desired_tile_width))
    logger.info('    expected_tile_shape %s' % str(expected_tile_shape))
    logger.info('    background_threshold %s' % str(background_threshold))
    logger.info('    background_pixel_value %s' % str(background_pixel_value))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Download SVS files using manifest and GDC extraction tool
    if do_download:
        if not os.path.exists(os.path.join(output_folder, 'has_been_downloaded')):
            if manifest_filepath is None or gdc_executable is None:
                raise ValueError('Download from TCGA is on: need both --gdc-executable and --manifest arguments filled')
            if not os.path.exists(gdc_executable):
                raise FileNotFoundError(f'Specified GDC executable {gdc_executable} not found')
            if not os.path.exists(manifest_filepath):
                raise FileNotFoundError(f'Specified manifest file {manifest_filepath} not found')

            crude_slides_output_folder = os.path.join(output_folder, 'downloaded_slides')
            if not os.path.exists(crude_slides_output_folder):
                os.makedirs(crude_slides_output_folder)

            logger.info(f'Downloading slides into {crude_slides_output_folder}...')
            slides_filepaths, md5sums, cases_ids = svs_factory.download_svs_files(gdc_executable, manifest_filepath,
                                                                                  crude_slides_output_folder)

            # Write control file after download is finished
            with open(os.path.join(output_folder, 'has_been_downloaded'), 'w') as f:
                f.write('\n'.join(','.join(a) for a in zip(slides_filepaths, md5sums, cases_ids)))
            logger.info('  done')
        else:
            logger.info('Slides already downloaded -> skipping')
            # Retrieve all downloaded SVS files in case previous step not performed
            with open(os.path.join(output_folder, 'has_been_downloaded'), 'r') as f:
                download_content = f.read().splitlines()
            download_content = list(map(lambda line: line.split(','), download_content))
            slides_filepaths, md5sums, cases_ids = list(map(list, zip(*download_content)))
    else:
        if source_slides_folder is None:
            raise ValueError('No download from TCGA: need --source-slides-folder argument filled with folder '
                             'containing slides to process')
        elif not os.path.exists(source_slides_folder):
            raise FileNotFoundError(f'Input folder {source_slides_folder} with slides to be processed not found')

        logger.info(f'Performing no download from TCGA as requested; listing source slides from {source_slides_folder}')
        slides_filepaths = svs_factory.list_slides_in_folder(source_slides_folder)
        # if not slide filepaths retrieved, try regime where slides are inside sup-folders
        if len(slides_filepaths) == 0:
            slides_filepaths = svs_factory.list_slides_in_folder(source_slides_folder, with_supfolder=True)
        md5sums = ['no_md5sum'] * len(slides_filepaths)
        cases_ids = ['no_case_id'] * len(slides_filepaths)

    # Tile all slides into super-patches
    has_been_tiled_filename = 'has_been_tiled_mag%d' % desired_magnification
    if not os.path.exists(os.path.join(output_folder, 'has_been_tiled_mag%d' % desired_magnification)):
        logger.info('Tiling slides into super-patches...')
        logger.info('  found %d files to be processed' % len(slides_filepaths))

        output_tiles_folders = svs_factory.tile_slides(slides_filepaths, desired_tile_width, desired_overlap,
                                                       desired_magnification)

        assert None not in output_tiles_folders
        assert len(output_tiles_folders) == len(slides_filepaths)

        # Write control file after tile processing is finished with tiles folders, slide names, md5sum and cases ID
        with open(os.path.join(output_folder, has_been_tiled_filename), 'w') as f:
            f.write('\n'.join(','.join(a) for a in zip(output_tiles_folders,
                                                       list(map(os.path.basename, slides_filepaths)),
                                                       md5sums, cases_ids)))
    else:
        logger.info('Slides already tiled at magnification %d -> skipping' % desired_magnification)

    # Independently from previous processing, extract labels of SVS files
    filtered_tiles_output_folder = os.path.join(output_folder, 'filtered_tiles')
    has_been_filtered_filename = os.path.join(output_folder, 'has_been_moved_and_filtered')
    if not os.path.exists(os.path.join(output_folder, 'has_been_moved_and_filtered')):
        logger.info('Extracting labels...')
        # Retrieve directories in which the super-patches are located in case previous step not performed
        with open(os.path.join(output_folder, has_been_tiled_filename), 'r') as f:
            tiled_content = f.read().splitlines()
        tiled_content = list(map(lambda line: line.split(','), tiled_content))
        output_tiles_folders, svs_filenames, md5sums, cases_ids = list(map(list, zip(*tiled_content)))

        associated_labels = list(map(case_factory.infer_class_from_tcga_name, svs_filenames))
        logger.info('  done')

        logger.info('Moving+background-filtering tiles into %s' % output_folder)
        data_folders = svs_factory.move_and_filter_tiles_folders(output_tiles_folders, associated_labels,
                                                                 svs_filenames, cases_ids,
                                                                 filtered_tiles_output_folder, background_pixel_value,
                                                                 background_threshold, expected_tile_shape,
                                                                 logger=logger)
        logger.info('  done')

        open(has_been_filtered_filename, 'w').write('\n'.join(data_folders))
        logger.info('Wrote `has_been_moved_and_filtered` file')
    else:
        logger.info('Tiles already moved and filtered -> skipping')
        # seek classes folders
        # data_folders = [f for f in os.listdir(filtered_tiles_output_folder)
        #                 if not os.path.isfile(os.path.join(filtered_tiles_output_folder, f))]
        data_folders = open(has_been_filtered_filename, 'r').read().splitlines()
        logger.info('Found %d source slides folders in %s' % (len(data_folders), filtered_tiles_output_folder))

        # logger.info('Performing train/val/test splitting with background removal')
        # train_cases_ids, val_cases_ids, test_cases_ids = case_factory.split_svs_samples_casewise(output_tiles_folders,
        #                                                                                          cases_ids,
        #                                                                                          train_size,
        #                                                                                          val_size,
        #                                                                                          test_size)

    logger.info('Pre-processing done')
    return list(map(lambda f: os.path.join(output_folder, f), data_folders))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    output_folder = args.output_folder
    gdc_executable = args.gdc_executable
    manifest_filepath = args.manifest
    do_download = not args.no_download
    source_slides_folder = args.source_slides_folder

    main(output_folder, do_download, gdc_executable, manifest_filepath, source_slides_folder)
