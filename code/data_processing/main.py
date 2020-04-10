__author__ = 'marvinler'

import os

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


def main(source_folder, output_folder, gdc_executable_path):
    logger = get_logger(filename_handler='data_processing.log', verbose=True)
    logger.info('Source folder %s' % os.path.abspath(source_folder) if source_folder else 'None')
    logger.info('Output folder %s' % os.path.abspath(output_folder))
    logger.info('Meta-parameters:')
    logger.info('  desired_magnification %s' % str(desired_magnification))
    logger.info('  tile_width %s' % str(desired_tile_width))
    logger.info('  expected_tile_shape %s' % str(expected_tile_shape))
    logger.info('  background_threshold %s' % str(background_threshold))
    logger.info('  background_pixel_value %s' % str(background_pixel_value))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Download SVS files using manifest and GDC extraction tool
    if not os.path.exists(os.path.join(output_folder, 'has_been_downloaded')):
        logger.info('Downloading slides...')
        svs_filepaths, md5sums, cases_ids = svs_factory.download_svs_files(source_folder, gdc_executable_path)

        # Write control file after download is finished
        with open(os.path.join(output_folder, 'has_been_downloaded'), 'w') as f:
            f.write('\n'.join(','.join(a) for a in zip(svs_filepaths, md5sums, cases_ids)))
        logger.info('  done')
    else:
        logger.info('Slides already downloaded -> skipping')

    # Tile all slides into super-patches
    if not os.path.exists(os.path.join(output_folder, 'has_been_tiled_mag%d' % desired_magnification)):
        logger.info('Tiling slides into super-patches...')
        # Retrieve all downloaded SVS files in case previous step not performed
        with open(os.path.join(output_folder, 'has_been_downloaded'), 'r') as f:
            download_content = f.read().splitlines()
        download_content = list(map(lambda line: line.split(','), download_content))
        svs_filepaths, md5sums, cases_ids = list(map(list, zip(*download_content)))
        logger.info('  found %d files to be processed' % len(svs_filepaths))

        output_tiles_folders = svs_factory.tile_slides(svs_filepaths, desired_tile_width, desired_overlap,
                                                       desired_magnification)

        assert None not in output_tiles_folders
        assert len(output_tiles_folders) == len(svs_filepaths)

        # Write control file after tile processing is finished with tiles folders, slide names, md5sum and cases ID
        with open(os.path.join(output_folder, 'has_been_tiled_mag%d' % desired_magnification), 'w') as f:
            f.write('\n'.join(','.join(a) for a in zip(output_tiles_folders,
                                                       list(map(os.path.basename, svs_filepaths)),
                                                       md5sums, cases_ids)))
    else:
        logger.info('Slides already tiled at magnification %d -> skipping' % desired_magnification)

    # Independently from previous processing, extract labels of SVS files
    if not os.path.exists(os.path.join(output_folder, 'has_been_moved_and_filtered')):
        logger.info('Extracting labels...')
        # Retrieve directories in which the super-patches are located in case previous step not performed
        with open(os.path.join(output_folder, 'has_been_tiled_mag%d' % desired_magnification), 'r') as f:
            tiled_content = f.read().splitlines()
        tiled_content = list(map(lambda line: line.split(','), tiled_content))
        output_tiles_folders, svs_filenames, md5sums, cases_ids = list(map(list, zip(*tiled_content)))

        associated_labels = list(map(case_factory.infer_class_from_tcga_name, svs_filenames))
        logger.info('  done')

        logger.info('Moving+background-filtering tiles into %s' % output_folder)
        data_folders = svs_factory.move_and_filter_tiles_folders(output_tiles_folders, associated_labels,
                                                                 svs_filenames, cases_ids,
                                                                 output_folder, background_pixel_value,
                                                                 background_threshold, expected_tile_shape,
                                                                 logger=logger)
        logger.info('  done')

        open(os.path.join(output_folder, 'has_been_moved_and_filtered'), 'w').write('\n'.join(data_folders))
        logger.info('Wrote `has_been_moved_and_filtered` file')
    else:
        logger.info('Tiles already moved and filtered -> skipping')
        # seek classes folders
        data_folders = [f for f in os.listdir(output_folder) if not os.path.isfile(os.path.join(output_folder, f))]
        logger.info('Found %d source slides folders' % len(data_folders))

        # logger.info('Performing train/val/test splitting with background removal')
        # train_cases_ids, val_cases_ids, test_cases_ids = case_factory.split_svs_samples_casewise(output_tiles_folders,
        #                                                                                          cases_ids,
        #                                                                                          train_size,
        #                                                                                          val_size,
        #                                                                                          test_size)

    return list(map(lambda f: os.path.join(output_folder, f), data_folders))


if __name__ == '__main__':
    source_folder = 'data/'
    output_folder = 'data/preprocessed/'
    gdc_client = os.path.join(source_folder, 'gdc-client')
    main(source_folder, output_folder, gdc_client)
