from dotenv import load_dotenv
load_dotenv()

from medvqa.utils.common import CACHE_DIR

import os

MIMICCXR_DATASET_DIR = os.environ['MIMICCXR_DATASET_DIR']
MIMICCXR_JPG_IMAGES_SMALL_DIR = os.environ['MIMICCXR_JPG_IMAGES_SMALL_DIR']
MIMICCXR_JPG_DIR = os.environ['MIMICCXR_JPG_DIR']
MIMICCXR_METADATA_CSV_PATH = os.path.join(MIMICCXR_JPG_DIR, 'mimic-cxr-2.0.0-metadata.csv')
MIMICCXR_SPLIT_CSV_PATH = os.path.join(MIMICCXR_JPG_DIR, 'mimic-cxr-2.0.0-split.csv')
# MIMICCXR_MEDICAL_TAGS_PATH = os.path.join(CACHE_DIR, 'mimiccxr', 'medical_tags_per_report.pkl')
MIMICCXR_CACHE_DIR = os.path.join(CACHE_DIR, 'mimiccxr')

# def preprocess_mimiccxr_data(self, tokenizer, mimiccxr_qa_reports, mimiccxr_metadata, mimiccxr_split):
        
#     self.report_ids = []
#     self.question_ids = []        
#     self.images = []
#     self.questions = []
#     self.answers = []
    
#     print('reading MIMIC-CXR splits ...')
    
#     split_dict = { (sub_id, stud_id, dicom_id) : split for sub_id, stud_id, dicom_id, split in zip(mimiccxr_split['subject_id'],
#                                                                                                     mimiccxr_split['study_id'],
#                                                                                                     mimiccxr_split['dicom_id'],
#                                                                                                     mimiccxr_split['split']) }        
#     print('reading MIMIC-CXR metadata ...')
    
#     image_views_dict = dict()
#     for subject_id, study_id, dicom_id, view_pos in zip(mimiccxr_metadata['subject_id'],
#                                                         mimiccxr_metadata['study_id'],
#                                                         mimiccxr_metadata['dicom_id'],
#                                                         mimiccxr_metadata['ViewPosition']):
#         key = (subject_id, study_id)
#         try:
#             views = image_views_dict[key]
#         except KeyError:
#             views = image_views_dict[key] = []
#         views.append((dicom_id, view_pos))
    
#     print('loading MIMIC-CXR vqa dataset ...')

#     broken_images = set()
#     for path in _MIMICCXR_BROKEN_IMAGES:
#         _, a, b, c = _MIMICCXR_IMAGE_REGEX.findall(path)[0]
#         broken_images.add((int(a), int(b), c))
    
#     question_list = mimiccxr_qa_reports['questions']
    
#     for ri, report in tqdm(enumerate(mimiccxr_qa_reports['reports'])):
        
#         sentences = report['sentences']
#         part_id, subject_id, study_id = map(int, _MIMICCXR_STUDY_REGEX.findall(report['filepath'])[0])
#         views = image_views_dict[(subject_id, study_id)]
#         # images = glob.glob(f'/mnt/workspace/mimic-cxr-jpg/images-small/p{part_id}/p{subject_id}/s{study_id}/*.jpg')
#         # assert len(views) == len(images)
        
#         dicom_id = None
#         if len(views) == 1:
#             dicom_id = views[0][0]
#         else:
#             for view in views:
#                 if view[1] == 'PA' or view[1] == 'AP':
#                     dicom_id = view[0]
#                     break
            
#         if (dicom_id and split_dict[(subject_id, study_id, dicom_id)] != 'test' and
#                 (subject_id, study_id, dicom_id) not in broken_images):
#             image_path = get_mimiccxr_image_path(part_id, subject_id, study_id, dicom_id)
#             for q_idx, a_idxs in report['qa'].items():
#                 q_idx = int(q_idx)
#                 question = question_list[q_idx]
#                 answer = ' '.join(sentences[i] for i in a_idxs)
#                 self.report_ids.append(ri)
#                 self.question_ids.append(q_idx)
#                 self.images.append(image_path)
#                 self.questions.append(tokenizer.string2ids(question.lower()))
#                 self.answers.append(tokenizer.string2ids(answer.lower()))