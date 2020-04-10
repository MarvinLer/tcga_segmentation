__author__ = 'marvinler'

import random


def infer_class_from_tcga_name(tcga_slide_filename):
    # https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/
    sample_code_and_vial = tcga_slide_filename.split('-')[3]
    sample_code = int(sample_code_and_vial[:2])
    assert 0 <= sample_code < 100

    # Return 1 for tumor type, 0 otherwise:
    # https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/sample-type-codes
    return int(sample_code < 10)


def split_svs_samples_casewise(svs_files, associated_cases_ids, val_size, test_size, seed=123):
    assert len(svs_files) == len(associated_cases_ids), 'Expected same number of SVS files than associated case ID'
    random.seed(seed)
    train_size = 1. - val_size - test_size

    unique_cases_ids = list(set(associated_cases_ids))
    random.shuffle(unique_cases_ids)
    total_unique_cases_ids = len(unique_cases_ids)

    # Extract cases ids for training, validation and testing sets
    train_cases_ids = unique_cases_ids[:int(train_size*total_unique_cases_ids)]
    val_cases_ids = unique_cases_ids[int(train_size*total_unique_cases_ids):
                                     int(train_size*total_unique_cases_ids)+int(val_size*total_unique_cases_ids)]
    test_cases_ids = unique_cases_ids[int(train_size*total_unique_cases_ids)+int(val_size*total_unique_cases_ids):]
    assert len(train_cases_ids) + len(val_cases_ids) + len(test_cases_ids) == total_unique_cases_ids

    # Compute associated split set for SVS files
    train_svs_files, val_svs_files, test_svs_files = [], [], []
    for svs_file, associated_case_id in zip(svs_files, associated_cases_ids):
        if associated_case_id in train_cases_ids:
            train_svs_files.append(svs_file)
        elif associated_case_id in val_cases_ids:
            val_svs_files.append(svs_file)
        else:
            test_svs_files.append(svs_file)

    return train_svs_files, val_svs_files, test_svs_files
