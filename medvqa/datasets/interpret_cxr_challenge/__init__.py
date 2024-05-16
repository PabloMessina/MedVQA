from dotenv import load_dotenv

from medvqa.utils.logging import print_red
load_dotenv()

from medvqa.datasets.chexpert import CHEXPERT_V1_0_SMALL_DATASET_DIR
from medvqa.datasets.iuxray import get_iuxray_image_path
from medvqa.datasets.mimiccxr import get_imageId2PartPatientStudy, get_imageId2reportId, get_mimiccxr_medium_image_path
from medvqa.datasets.padchest import PADCHEST_IMAGES_SMALL_DIR
from medvqa.utils.files import get_cached_dataframe_from_csv, get_cached_json_file

import os
import numpy as np
import pandas as pd
from collections import Counter

INTERPRET_CXR_TEST_PUBLIC_IMAGES_FOLDER_PATH = os.environ['INTERPRET_CXR_TEST_PUBLIC_IMAGES_FOLDER_PATH']
INTERPRET_CXR_TEST_PUBLIC_CSV_PATH = os.environ['INTERPRET_CXR_TEST_PUBLIC_CSV_PATH']
INTERPRET_CXR_TEST_HIDDEN_IMAGES_FOLDER_PATH = os.environ['INTERPRET_CXR_TEST_HIDDEN_IMAGES_FOLDER_PATH']
INTERPRET_CXR_TEST_HIDDEN_CSV_PATH = os.environ['INTERPRET_CXR_TEST_HIDDEN_CSV_PATH']
INTERPRET_CXR_TRAIN_CSV_PATH = os.environ['INTERPRET_CXR_TRAIN_CSV_PATH']
INTERPRET_CXR_VAL_CSV_PATH = os.environ['INTERPRET_CXR_VAL_CSV_PATH']
INTERPRET_CXR_TRAIN_MIMICCXR_JSON_FILEPATH = os.environ['INTERPRET_CXR_TRAIN_MIMICCXR_JSON_FILEPATH']
INTERPRET_CXR_VAL_MIMICCXR_JSON_FILEPATH = os.environ['INTERPRET_CXR_VAL_MIMICCXR_JSON_FILEPATH']
BIMCV_COVID19_IMAGES_DIR = os.environ['BIMCV_COVID19_IMAGES_DIR']

class BaseImageToReportMapper:
    def __init__(self):
        self._image_path_to_report_id = {}
        self._findings_list = []
        self._impression_list = []
        self._splits = []
        self._image_paths_list = []

    def get_impression(self, image_path):
        return self._impression_list[self._image_path_to_report_id[image_path]]
    
    def get_findings(self, image_path):
        return self._findings_list[self._image_path_to_report_id[image_path]]
    
    def sanity_check(self):
        for i, image_paths in enumerate(self._image_paths_list):
            for ip in image_paths:
                assert self._image_path_to_report_id[ip] == i

    def get_sections_and_image_indexes_list(self, section, image_path_to_index):
        assert section in ['findings', 'impression', 'both']
        sections_list = []
        image_indexes_list = []
        skipped_images = 0
        skipped_reports_because_of_no_images = 0
        skipped_reports_because_of_no_section = 0
        for i, image_paths in enumerate(self._image_paths_list):
            if section == 'findings':
                section_text = self._findings_list[i]
            elif section == 'impression':
                section_text = self._impression_list[i]
            elif section == 'both':
                section_text = ''
                if self._findings_list[i]:
                    section_text += self._findings_list[i]
                if self._impression_list[i]:
                    if section_text:
                        if section_text[-1] == '.':
                            section_text += ' '
                        else:
                            section_text += '. '
                    section_text += self._impression_list[i]
            else:
                raise ValueError(f'Invalid section: {section}')
            if section_text:
                image_indexes = [image_path_to_index[ip] for ip in image_paths if ip in image_path_to_index]
                skipped_images += len(image_paths) - len(image_indexes)
                if not image_indexes:
                    skipped_reports_because_of_no_images += 1
                    continue
                sections_list.append(section_text)
                image_indexes_list.append(image_indexes)
            else:
                skipped_reports_because_of_no_section += 1
        if skipped_images > 0:
            print_red(f'skipped_images = {skipped_images}', bold=True)
        if skipped_reports_because_of_no_images > 0:
            print_red(f'skipped_reports_because_of_no_images = {skipped_reports_because_of_no_images}', bold=True)
        if skipped_reports_because_of_no_section > 0:
            print_red(f'skipped_reports_because_of_no_section = {skipped_reports_because_of_no_section}', bold=True)
        return sections_list, image_indexes_list

class MIMICCXR_ImageToReportMapper(BaseImageToReportMapper):
    def __init__(self, background_findings_and_impression_per_report_filepath):
        super().__init__()

        mimimiccxr_train_data = get_cached_json_file(INTERPRET_CXR_TRAIN_MIMICCXR_JSON_FILEPATH)
        mimimiccxr_val_data = get_cached_json_file(INTERPRET_CXR_VAL_MIMICCXR_JSON_FILEPATH)
        mimbfipr = get_cached_json_file(background_findings_and_impression_per_report_filepath)
        imageId2PartPatientStudy = get_imageId2PartPatientStudy()
        imageId2reportId = get_imageId2reportId()
        
        for item in mimimiccxr_train_data:
            image_paths = []
            rids = []
            for image_path in item['images_path']:
                dicom_id = os.path.basename(image_path).split('.')[0] # remove extension
                image_path = get_mimiccxr_medium_image_path(*imageId2PartPatientStudy[dicom_id], dicom_id)
                assert os.path.exists(image_path)
                image_paths.append(image_path)
                rid = imageId2reportId[dicom_id]
                rids.append(rid)
                assert image_path not in self._image_path_to_report_id # no duplicate image paths
                self._image_path_to_report_id[image_path] = len(self._findings_list)
            assert all(rids[0] == rid for rid in rids) # all images in the same item should have the same report id
            rid = rids[0]
            self._findings_list.append(mimbfipr[rid]['findings'])
            self._impression_list.append(mimbfipr[rid]['impression'])
            self._image_paths_list.append(image_paths)
            self._splits.append('train')
        
        for item in mimimiccxr_val_data:
            image_paths = []
            rids = []
            for image_path in item['images_path']:
                dicom_id = os.path.basename(image_path).split('.')[0]
                image_path = get_mimiccxr_medium_image_path(*imageId2PartPatientStudy[dicom_id], dicom_id)
                assert os.path.exists(image_path)
                image_paths.append(image_path)
                rid = imageId2reportId[dicom_id]
                rids.append(rid)
                assert image_path not in self._image_path_to_report_id # no duplicate image paths
                self._image_path_to_report_id[image_path] = len(self._findings_list)
            assert all(rids[0] == rid for rid in rids)
            rid = rids[0]
            self._findings_list.append(mimbfipr[rid]['findings'])
            self._impression_list.append(mimbfipr[rid]['impression'])
            self._image_paths_list.append(image_paths)
            self._splits.append('val')

        print(Counter(self._splits))

        # Sanity check
        self.sanity_check()

class OpenI_ImageToReportMapper(BaseImageToReportMapper):
    def __init__(self):
        super().__init__()

        challenge_train_df = get_cached_dataframe_from_csv(INTERPRET_CXR_TRAIN_CSV_PATH)
        challenge_val_df = get_cached_dataframe_from_csv(INTERPRET_CXR_VAL_CSV_PATH)
        challenge_train_df.fillna('', inplace=True)
        challenge_val_df.fillna('', inplace=True)
        openi_train_df = challenge_train_df[challenge_train_df['source'] == 'OpenI']
        openi_val_df = challenge_val_df[challenge_val_df['source'] == 'OpenI']

        assert len(openi_train_df) > 0
        assert len(openi_val_df) > 0

        for images_path_old, findings, impression in\
                openi_train_df[['images_path_old', 'findings', 'impression']].values:
            images_path_old = [eval(x) for x in images_path_old[1:-1].split()]
            actual_image_paths = []
            for image_path in images_path_old:
                image_id = os.path.basename(image_path).split('.')[0]
                actual_image_path = get_iuxray_image_path(image_id)
                assert os.path.exists(actual_image_path)
                assert actual_image_path not in self._image_path_to_report_id
                self._image_path_to_report_id[actual_image_path] = len(self._findings_list)
                actual_image_paths.append(actual_image_path)
            self._findings_list.append(findings)
            self._impression_list.append(impression)
            self._image_paths_list.append(actual_image_paths)
            self._splits.append('train')

        for images_path_old, findings, impression in\
                openi_val_df[['images_path_old', 'findings', 'impression']].values:
            images_path_old = [eval(x) for x in images_path_old[1:-1].split()]
            actual_image_paths = []
            for image_path in images_path_old:
                image_id = os.path.basename(image_path).split('.')[0]
                actual_image_path = get_iuxray_image_path(image_id)
                assert os.path.exists(actual_image_path)
                assert actual_image_path not in self._image_path_to_report_id
                self._image_path_to_report_id[actual_image_path] = len(self._findings_list)
                actual_image_paths.append(actual_image_path)
            self._findings_list.append(findings)
            self._impression_list.append(impression)
            self._image_paths_list.append(actual_image_paths)
            self._splits.append('val')

        print(Counter(self._splits))
        
        # Sanity check
        self.sanity_check()
    
class CheXpert_ImageToReportMapper(BaseImageToReportMapper):
    def __init__(self):
        super().__init__()

        challenge_train_df = get_cached_dataframe_from_csv(INTERPRET_CXR_TRAIN_CSV_PATH)
        challenge_val_df = get_cached_dataframe_from_csv(INTERPRET_CXR_VAL_CSV_PATH)
        challenge_train_df.fillna('', inplace=True)
        challenge_val_df.fillna('', inplace=True)
        chexpert_train_df = challenge_train_df[challenge_train_df['source'] == 'CheXpert']
        chexpert_val_df = challenge_val_df[challenge_val_df['source'] == 'CheXpert']

        assert len(chexpert_train_df) > 0
        assert len(chexpert_val_df) > 0

        for images_path_old, findings, impression in\
                chexpert_train_df[['images_path_old', 'findings', 'impression']].values:
            images_path_old = [eval(x)[21:] for x in images_path_old[1:-1].split()]
            actual_image_paths = []
            for image_path in images_path_old:
                image_path = os.path.join(CHEXPERT_V1_0_SMALL_DATASET_DIR, image_path)
                assert os.path.exists(image_path)
                assert image_path not in self._image_path_to_report_id
                self._image_path_to_report_id[image_path] = len(self._findings_list)
                actual_image_paths.append(image_path)
            self._findings_list.append(findings)
            self._impression_list.append(impression)
            self._image_paths_list.append(actual_image_paths)
            self._splits.append('train')

        for images_path_old, findings, impression in\
                chexpert_val_df[['images_path_old', 'findings', 'impression']].values:
            images_path_old = [eval(x)[21:] for x in images_path_old[1:-1].split()]
            actual_image_paths = []
            for image_path in images_path_old:
                image_path = os.path.join(CHEXPERT_V1_0_SMALL_DATASET_DIR, image_path)
                assert os.path.exists(image_path)
                assert image_path not in self._image_path_to_report_id
                self._image_path_to_report_id[image_path] = len(self._findings_list)
                actual_image_paths.append(image_path)
            self._findings_list.append(findings)
            self._impression_list.append(impression)
            self._image_paths_list.append(actual_image_paths)
            self._splits.append('val')

        print(Counter(self._splits))
        
        # Sanity check
        self.sanity_check()

class PadChest_ImageToReportMapper(BaseImageToReportMapper):
    def __init__(self):
        super().__init__()

        challenge_train_df = get_cached_dataframe_from_csv(INTERPRET_CXR_TRAIN_CSV_PATH)
        challenge_val_df = get_cached_dataframe_from_csv(INTERPRET_CXR_VAL_CSV_PATH)
        challenge_train_df.fillna('', inplace=True)
        challenge_val_df.fillna('', inplace=True)
        padchest_train_df = challenge_train_df[challenge_train_df['source'] == 'PadChest']
        padchest_val_df = challenge_val_df[challenge_val_df['source'] == 'PadChest']

        assert len(padchest_train_df) > 0
        assert len(padchest_val_df) > 0

        for images_path_old, findings, impression in\
                padchest_train_df[['images_path_old', 'findings', 'impression']].values:
            images_path_old = [eval(x) for x in images_path_old[1:-1].split()]
            actual_image_paths = []
            for image_path in images_path_old:
                actual_image_path = os.path.join(PADCHEST_IMAGES_SMALL_DIR, os.path.basename(image_path))
                assert os.path.exists(actual_image_path)
                assert actual_image_path not in self._image_path_to_report_id
                self._image_path_to_report_id[actual_image_path] = len(self._findings_list)
                actual_image_paths.append(actual_image_path)
            self._findings_list.append(findings)
            self._impression_list.append(impression)
            self._image_paths_list.append(actual_image_paths)
            self._splits.append('train')

        for images_path_old, findings, impression in\
                padchest_val_df[['images_path_old', 'findings', 'impression']].values:
            images_path_old = [eval(x) for x in images_path_old[1:-1].split()]
            actual_image_paths = []
            for image_path in images_path_old:
                actual_image_path = os.path.join(PADCHEST_IMAGES_SMALL_DIR, os.path.basename(image_path))
                assert os.path.exists(actual_image_path)
                assert actual_image_path not in self._image_path_to_report_id
                self._image_path_to_report_id[actual_image_path] = len(self._findings_list)
                actual_image_paths.append(actual_image_path)
            self._findings_list.append(findings)
            self._impression_list.append(impression)
            self._image_paths_list.append(actual_image_paths)
            self._splits.append('val')

        print(Counter(self._splits))

        # Sanity check
        self.sanity_check()

class BIMCV_COVID19_ImageToReportMapper(BaseImageToReportMapper):
    def __init__(self):
        super().__init__()

        challenge_train_df = get_cached_dataframe_from_csv(INTERPRET_CXR_TRAIN_CSV_PATH)
        challenge_val_df = get_cached_dataframe_from_csv(INTERPRET_CXR_VAL_CSV_PATH)
        challenge_train_df.fillna('', inplace=True)
        challenge_val_df.fillna('', inplace=True)
        bimcv_train_df = challenge_train_df[challenge_train_df['source'] == 'BIMCV-COVID19']
        bimcv_val_df = challenge_val_df[challenge_val_df['source'] == 'BIMCV-COVID19']

        assert len(bimcv_train_df) > 0
        assert len(bimcv_val_df) > 0

        self._image_path_to_report_id = {}
        self._findings_list = []
        self._impression_list = []
        self._splits = []
        self._image_paths_list = []

        for images_path, findings, impression in bimcv_train_df[['images_path', 'findings', 'impression']].values:
            images_path = eval(images_path)
            actual_image_paths = []
            for image_path in images_path:
                actual_image_path = os.path.join(BIMCV_COVID19_IMAGES_DIR, os.path.basename(image_path))
                assert os.path.exists(actual_image_path)
                assert actual_image_path not in self._image_path_to_report_id
                self._image_path_to_report_id[actual_image_path] = len(self._findings_list)
                actual_image_paths.append(actual_image_path)
            self._findings_list.append(findings)
            self._impression_list.append(impression)
            self._image_paths_list.append(actual_image_paths)
            self._splits.append('train')

        collision_count = 0

        for images_path, findings, impression in bimcv_val_df[['images_path', 'findings', 'impression']].values:
            images_path = eval(images_path)
            actual_image_paths = []
            for image_path in images_path:
                actual_image_path = os.path.join(BIMCV_COVID19_IMAGES_DIR, os.path.basename(image_path))
                assert os.path.exists(actual_image_path)
                try:
                    assert actual_image_path not in self._image_path_to_report_id
                    self._image_path_to_report_id[actual_image_path] = len(self._findings_list)
                except AssertionError:
                    collision_count += 1
                actual_image_paths.append(actual_image_path)
            self._findings_list.append(findings)
            self._impression_list.append(impression)
            self._image_paths_list.append(actual_image_paths)
            self._splits.append('val')

        if collision_count > 0:
            print_red(f'collision_count = {collision_count}', bold=True)

        print(Counter(self._splits))


def load_interpret_cxr_test_public_data(section, csv_with_source_predictions_filepath=None):
    assert section in ['findings', 'impression', 'both']
    if csv_with_source_predictions_filepath is not None:
        print(f'Using csv_with_source_predictions_filepath: {csv_with_source_predictions_filepath}')
        return_source_predictions = True
        df = pd.read_csv(csv_with_source_predictions_filepath)
    else: # use the default csv file
        return_source_predictions = False
        df = pd.read_csv(INTERPRET_CXR_TEST_PUBLIC_CSV_PATH)
    df = df.replace(np.nan, '', regex=True) # replace nan with empty string
    image_paths_list = df['images_path'].tolist()
    findings_list = df['findings'].tolist()
    impression_list = df['impression'].tolist()
    image_paths_list = [eval(x) for x in image_paths_list] # convert string to list
    image_paths_list_ = []
    gt_reports_ = []
    if return_source_predictions:
        source_predictions = df['source_pred'].tolist()
        source_predictions_ = []

    for i, image_paths in enumerate(image_paths_list):
        
        if section == 'findings':
            gt_report = findings_list[i]
        elif section == 'impression':
            gt_report = impression_list[i]
        elif section == 'both':
            gt_report = ''
            if findings_list[i]:
                gt_report += findings_list[i]
            if impression_list[i]:
                if gt_report:
                    if gt_report[-1] == '.':
                        gt_report += ' '
                    else:
                        gt_report += '. '
                gt_report += impression_list[i]
        else:
            raise ValueError(f'Invalid section: {section}')
        
        if gt_report:
            # report
            gt_reports_.append(gt_report)
            # image paths
            image_paths_ = []
            for image_path in image_paths:
                image_path = os.path.join(INTERPRET_CXR_TEST_PUBLIC_IMAGES_FOLDER_PATH, os.path.basename(image_path))
                assert os.path.exists(image_path)
                image_paths_.append(image_path)
            image_paths_list_.append(image_paths_)
            # source predictions
            if return_source_predictions:
                source_predictions_.append(source_predictions[i])

    if return_source_predictions:
        return image_paths_list_, gt_reports_, source_predictions_
    return image_paths_list_, gt_reports_

def load_interpret_cxr_test_public_image_paths():
    df = pd.read_csv(INTERPRET_CXR_TEST_PUBLIC_CSV_PATH)
    df = df.replace(np.nan, '', regex=True) # replace nan with empty string
    image_paths_list = df['images_path'].tolist()
    image_paths_list = [eval(x) for x in image_paths_list] # convert string to list
    image_path_list = []
    for image_paths in image_paths_list:
        for image_path in image_paths:
            image_path = os.path.join(INTERPRET_CXR_TEST_PUBLIC_IMAGES_FOLDER_PATH, os.path.basename(image_path))
            assert os.path.exists(image_path)
            image_path_list.append(image_path)
    return image_path_list

def load_interpret_cxr_test_hidden_image_paths():
    df = pd.read_csv(INTERPRET_CXR_TEST_HIDDEN_CSV_PATH)
    df = df.replace(np.nan, '', regex=True) # replace nan with empty string
    image_paths_list = df['images_path'].tolist()
    image_paths_list = [eval(x) for x in image_paths_list] # convert string to list
    image_path_list = []
    for image_paths in image_paths_list:
        for image_path in image_paths:
            image_path = os.path.join(INTERPRET_CXR_TEST_HIDDEN_IMAGES_FOLDER_PATH, os.path.basename(image_path))
            assert os.path.exists(image_path)
            image_path_list.append(image_path)
    return image_path_list
