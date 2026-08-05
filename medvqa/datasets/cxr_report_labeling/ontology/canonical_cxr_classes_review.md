# Canonical CXR ontology — current class set

Source of truth: `canonical_cxr_ontology.yaml` (schema 1.0.0, ontology version 1.0.0, status: review).

Image-oriented multilabel ontology harmonizing the current canonical CXR classes with CXR-LT 2024, PUCCXR, and VinDr-CXR.

## Design principles

- Parent edges are true semantic is-a / label-inheritance: a positive child propagates to every parent. Organizational grouping uses `category` only.
- Prefer image-grounded morphology over etiology when CXR alone cannot establish the latter.
- Retain high-value diagnostic labels (`Pneumonia`, `Tuberculosis`, `Interstitial lung disease`) as nonexclusive supervised targets, separate from morphology.
- Umbrella classifier labels (`Lung opacity`, `Calcification`, `Enlarged cardiomediastinal silhouette`, `Bone fracture`, `Support device present`, `Hernia`) are real targets and receive propagation from children.
- Merge distinctions unreliable on CXR (e.g., nodule vs mass size cutoff; cyst vs bulla; eventration vs elevation; fissural vs other loculated effusion).
- `No finding` is an image-level state requiring adequate quality and absence of every modeled abnormality or support device.

## Mapping legend

Exact/near matches are written without qualification when `relationship: exact`, or as `(near)`. Non-equivalent overlap uses `(broader)`, `(narrower)`, or `(partial)`. An em dash means no useful match.

## Classes (79 total)

| # | Class | Category | Parents | Kind | Def .txt | PUC-CXR | CXR-LT 2024 | VinDr-CXR |
|---:|---|---|---|---|:---:|---|---|---|
| 1 | No finding | Global | — | state | Yes | — | Normal (near) | No finding |
| 2 | Calcification | Calcific findings | — | finding | Yes | Calcified granuloma (narrower); Calcified heart valve (narrower); Pleural plaques (narrower) | Calcification of the Aorta (narrower) | Calcification |
| 3 | Lung opacity | Pulmonary opacity | — | finding | Yes | Ground glass pattern (narrower) | Lung Opacity; Infiltration (partial) | Lung Opacity; Infiltration (partial) |
| 4 | Ground-glass opacity/pattern | Pulmonary opacity | Lung opacity | finding | Yes | Ground glass pattern | Lung Opacity (broader); Infiltration (partial) | Lung Opacity (broader); Infiltration (partial) |
| 5 | Consolidation | Pulmonary opacity | Lung opacity | finding | Yes | Consolidation | Consolidation | Consolidation |
| 6 | Pneumonia | Diagnostic label | Lung opacity | diagnosis | Yes | Pneumonia | Pneumonia | Pneumonia |
| 7 | Tuberculosis | Diagnostic label | — | diagnosis | Yes | — | Tuberculosis (near) | Tuberculosis (near) |
| 8 | Interstitial lung disease | Diagnostic label | — | diagnosis | Yes | Reticular interstitial pattern (partial); Pulmonary fibrosis (narrower) | Fibrosis (narrower) | ILD |
| 9 | Atelectasis | Pulmonary opacity | Lung opacity | finding | Yes | Atelectasis | Atelectasis | Atelectasis |
| 10 | Lobar atelectasis | Pulmonary opacity | Atelectasis | finding | Yes | Atelectasis (broader) | Lobar Atelectasis | Atelectasis (broader) |
| 11 | Rounded atelectasis | Pulmonary opacity | Atelectasis | finding | Yes | Atelectasis (broader) | Round(ed) Atelectasis | Atelectasis (broader) |
| 12 | Reticular interstitial opacity/pattern | Pulmonary opacity | Lung opacity | finding | Yes | Reticular interstitial pattern | Fibrosis (partial); Infiltration (partial) | ILD (partial); Pulmonary fibrosis (partial) |
| 13 | Kerley lines | Pulmonary opacity | Lung opacity | finding | Yes | Kerley lines | Edema (partial) | Edema (partial) |
| 14 | Pulmonary edema | Pulmonary opacity | Lung opacity | finding | Yes | Edema (near); Kerley lines (narrower) | Edema (near) | Edema (near) |
| 15 | Fibrotic lung change | Pulmonary opacity | Lung opacity | finding | Yes | Pulmonary fibrosis (near) | Fibrosis (near) | Pulmonary fibrosis (near); ILD (partial) |
| 16 | Focal pulmonary scarring | Pulmonary opacity | Lung opacity | finding | Yes | Pulmonary fibrosis (broader) | Fibrosis (broader) | Pulmonary fibrosis (broader) |
| 17 | Pulmonary nodule or mass | Focal lesion findings | Lung opacity | finding | Yes | Nodule; Lung Mass; Granuloma (partial) | Nodule; Mass; Lung Lesion (near) | Nodule/Mass; Lung tumor (partial) |
| 18 | Calcified pulmonary nodule or granuloma | Focal lesion findings | Calcification, Pulmonary nodule or mass | finding | Yes | Calcified granuloma (near); Granuloma (partial) | Granuloma (near) | Calcification (broader) |
| 19 | Pulmonary cavity | Focal lesion findings | — | finding | Yes | Cavitation (near) | Tuberculosis (partial) | Lung cavity; Tuberculosis (partial) |
| 20 | Pulmonary cyst or bulla | Focal lesion findings | — | finding | Yes | Bulla | Bulla | Lung cyst (near); Emphysema (partial) |
| 21 | Emphysema | Pulmonary parenchyma | — | finding | Yes | — | Emphysema | Emphysema; COPD (partial) |
| 22 | Bronchiectasis | Airway | — | finding | Yes | Bronchiectasis | — | — |
| 23 | Bronchial wall thickening | Airway | — | finding | Yes | — | — | — |
| 24 | Low lung volume | Lung volume | — | finding | Yes | — | — | — |
| 25 | Hyperinflation | Lung volume | — | finding | Yes | Hyperinflated lung (near); Flattened diaphragm (partial) | Emphysema (partial) | COPD (partial); Emphysema (partial) |
| 26 | Pleural effusion | Pleura | — | finding | Yes | Pleural effusion | Pleural Effusion | Pleural effusion |
| 27 | Loculated/fissural effusion | Pleura | Pleural effusion | finding | Yes | Pleural effusion (broader) | Fissure (partial); Pleural Effusion (broader) | Pleural effusion (broader) |
| 28 | Costophrenic angle blunting | Pleura | — | finding | Yes | Costophrenic angle blunting | Pleural Effusion (partial); Pleural Other (partial) | Pleural effusion (partial) |
| 29 | Pneumothorax | Pleura | — | finding | Yes | Pneumothorax | Pneumothorax | Pneumothorax |
| 30 | Hydropneumothorax | Pleura | Pleural effusion, Pneumothorax | finding | Yes | Pleural effusion (partial); Pneumothorax (partial) | Hydropneumothorax | Pleural effusion (partial); Pneumothorax (partial) |
| 31 | Pleural thickening | Pleura | — | finding | Yes | Pleural thickening | Pleural Thickening | Pleural thickening |
| 32 | Pleural plaque | Pleura | Pleural thickening | finding | Yes | Pleural plaques | Pleural Other (broader) | Pleural thickening (partial); Calcification (broader) |
| 33 | Pleural calcification | Pleura | Pleural thickening, Calcification | finding | Yes | Pleural plaques (partial) | Pleural Other (broader) | Calcification (broader) |
| 34 | Pleural nodule or mass | Pleura | — | finding | Yes | Pleural mass (near) | Pleural Other (broader); Mass (partial) | Other lesion (broader) |
| 35 | Enlarged cardiomediastinal silhouette | Cardiomediastinal | — | finding | Yes | Cardiomegaly (narrower); Mediastinal enlargement (narrower) | Enlarged Cardiomediastinum | Cardiomegaly (narrower); Aortic enlargement (narrower) |
| 36 | Enlarged cardiac silhouette | Cardiomediastinal | Enlarged cardiomediastinal silhouette | finding | Yes | Cardiomegaly (near) | Cardiomegaly (near); Enlarged Cardiomediastinum (broader) | Cardiomegaly (near) |
| 37 | Pulmonary vascular congestion | Cardiomediastinal | — | finding | Yes | Vascular redistribution (near) | Edema (partial) | Edema (partial) |
| 38 | Enlarged pulmonary artery | Cardiomediastinal | — | finding | Yes | — | Pulmonary Hypertension (partial); Hilum (partial) | Enlarged PA |
| 39 | Aortic enlargement/ectasia | Cardiomediastinal | Enlarged cardiomediastinal silhouette | finding | Yes | Aortic elongation (partial) | Enlarged Cardiomediastinum (partial) | Aortic enlargement |
| 40 | Aortic tortuosity/elongation | Cardiomediastinal | — | finding | Yes | Aortic elongation | Tortuous Aorta | Aortic enlargement (partial) |
| 41 | Aortic calcification | Cardiomediastinal | Calcification | finding | Yes | Aortic atheromatosis (partial) | Calcification of the Aorta | Calcification (broader) |
| 42 | Cardiac/valvular/pericardial calcification | Cardiomediastinal | Calcification | finding | Yes | Calcified heart valve (narrower) | — | Calcification (broader) |
| 43 | Hilar enlargement | Cardiomediastinal | — | finding | Yes | Hilar enlargement | Hilum (partial); Adenopathy (partial) | — |
| 44 | Hilar or mediastinal nodal enlargement | Cardiomediastinal | — | finding | Yes | Hilar enlargement (partial); Mediastinal enlargement (partial) | Adenopathy (near) | — |
| 45 | Mediastinal widening | Cardiomediastinal | Enlarged cardiomediastinal silhouette | finding | Yes | Mediastinal enlargement (near) | Enlarged Cardiomediastinum (broader) | — |
| 46 | Mediastinal mass | Cardiomediastinal | — | finding | Yes | Mediastinal mass | Mass (partial); Adenopathy (partial) | Other lesion (broader) |
| 47 | Mediastinal shift | Cardiomediastinal | — | finding | Yes | — | — | Mediastinal shift |
| 48 | Pneumomediastinum | Cardiomediastinal | — | finding | Yes | — | Pneumomediastinum | — |
| 49 | Tracheal deviation | Airway | — | finding | Yes | — | — | Mediastinal shift (partial) |
| 50 | Elevated/eventrated hemidiaphragm | Diaphragm | — | finding | Yes | Hemidiaphragm elevation | — | — |
| 51 | Flattened diaphragm | Diaphragm | — | attribute | Yes | Flattened diaphragm | Emphysema (partial) | COPD (partial); Emphysema (partial) |
| 52 | Hernia | Diaphragm | — | finding | Yes | Hiatal hernia (narrower) | Hernia | — |
| 53 | Hiatal hernia | Diaphragm | Hernia | finding | Yes | Hiatal hernia | Hernia (broader) | — |
| 54 | Pneumoperitoneum | Upper abdomen | — | finding | Yes | — | Pneumoperitoneum | — |
| 55 | Bone fracture | Bone fracture findings | — | finding | Yes | Fracture | Fracture | Rib fracture (narrower); Clavicle fracture (narrower) |
| 56 | Clavicle fracture | Bone fracture findings | Bone fracture | finding | Yes | Fracture (broader) | Clavicle Fracture | Clavicle fracture |
| 57 | Rib fracture | Bone fracture findings | Bone fracture | finding | Yes | Rib fracture | Rib Fracture | Rib fracture |
| 58 | Vertebral compression fracture | Bone fracture findings | Bone fracture | finding | Yes | Fracture (broader) | Fracture (broader) | — |
| 59 | Focal osseous lesion | Musculoskeletal | — | finding | Yes | Bone lesion (near) | Fracture (partial) | Other lesion (broader) |
| 60 | Degenerative osseous change | Musculoskeletal | — | finding | Yes | Degenerative changes (near) | — | — |
| 61 | Radiographic osteopenia | Musculoskeletal | — | finding | Yes | Osteopenia (near) | Osteopenia (near) | — |
| 62 | Scoliosis | Musculoskeletal | — | finding | Yes | Scoliosis | Scoliosis | — |
| 63 | Kyphosis | Musculoskeletal | — | finding | Yes | — | Kyphosis | — |
| 64 | Support device present | Support device | — | finding | Yes | Lines (narrower); Tubes (narrower); Pacemaker (narrower); Artificial heart valve (narrower) | Support Devices | — |
| 65 | Support line present | Support device | Support device present | finding | Yes | Lines | Support Devices (broader) | — |
| 66 | Support tube present | Support device | Support device present | finding | Yes | Tubes | Support Devices (broader) | — |
| 67 | Endotracheal tube | Support device | Support tube present | device | Yes | Tubes (broader) | Support Devices (broader) | — |
| 68 | Enteric tube | Support device | Support tube present | device | Yes | Tubes (broader) | Support Devices (broader) | — |
| 69 | Central venous catheter | Support device | Support line present | device | Yes | Lines (broader) | Support Devices (broader) | — |
| 70 | Chest tube | Support device | Support tube present | device | Yes | Tubes (broader) | Support Devices (broader) | — |
| 71 | Pacemaker/ICD | Support device | Support device present | device | Yes | Pacemaker (narrower) | Support Devices (broader) | — |
| 72 | Prosthetic cardiac valve | Support device | Support device present | device | Yes | Artificial heart valve (near) | Support Devices (broader) | — |
| 73 | Vascular graft/stent | Support device | Support device present | device | Yes | Aortic endoprosthesis (narrower) | Support Devices (broader) | — |
| 74 | Breast implant | Support device | Support device present | device | Yes | — | — | — |
| 75 | Sternotomy wires | Support device | Support device present | device | Yes | Sternotomy (near) | Support Devices (broader) | — |
| 76 | Device abnormality | Support device | Support device present | attribute | Yes | Lines (partial); Tubes (partial) | Support Devices (partial) | — |
| 77 | Subcutaneous emphysema | Soft tissue | — | finding | Yes | Subcutaneous emphysema | Subcutaneous Emphysema | — |
| 78 | Nontherapeutic foreign body | Other | — | finding | Yes | Abnormal foreign body (near) | — | Other lesion (broader) |
| 79 | Azygos lobe | Anatomic variant | — | finding | Yes | — | Azygos Lobe | — |

## Hierarchy by category (is-a edges)

### Global

Image-level states and globally distributed findings.

- **No finding** (`no_finding`)

### Calcific findings

Organizational group for thoracic calcific findings modeled under the Calcification umbrella class.

- **Calcification** (`calcification`)

### Pulmonary opacity

Organizational group for pulmonary opacity patterns and related radiographic syndromes modeled under the Lung opacity umbrella class.

- **Lung opacity** (`lung_opacity`)
  - **Atelectasis** (`atelectasis`)
    - **Lobar atelectasis** (`lobar_atelectasis`)
    - **Rounded atelectasis** (`rounded_atelectasis`)
  - **Consolidation** (`consolidation`)
  - **Fibrotic lung change** (`fibrotic_lung_change`)
  - **Focal pulmonary scarring** (`focal_pulmonary_scarring`)
  - **Ground-glass opacity/pattern** (`ground_glass_opacity_pattern`)
  - **Kerley lines** (`kerley_lines`)
  - **Pulmonary edema** (`pulmonary_edema`)
  - **Reticular interstitial opacity/pattern** (`reticular_interstitial_opacity_pattern`)

### Diagnostic label

Explicit supervised diagnostic labels retained separately from image morphology.

- **Interstitial lung disease** (`interstitial_lung_disease`)
- **Pneumonia** (`pneumonia`) — parents: Lung opacity
- **Tuberculosis** (`tuberculosis`)

### Focal lesion findings

Organizational group for discrete intrapulmonary nodules/masses, cavities, and thin-walled cystic or bullous lucencies.

- **Pulmonary cavity** (`pulmonary_cavity`)
- **Pulmonary cyst or bulla** (`pulmonary_cyst_or_bulla`)
- **Pulmonary nodule or mass** (`pulmonary_nodule_or_mass`) — parents: Lung opacity
  - **Calcified pulmonary nodule or granuloma** (`calcified_pulmonary_nodule_or_granuloma`) — parents: Calcification

### Pulmonary parenchyma

Pulmonary findings not appropriately organized as opacity or focal-lesion subtypes.

- **Emphysema** (`emphysema`)

### Lung volume

Findings describing reduced or increased lung volume.

- **Hyperinflation** (`hyperinflation`)
- **Low lung volume** (`low_lung_volume`)

### Pleura

Pleural fluid, gas, thickening, plaque, calcification, and focal pleural lesions.

- **Costophrenic angle blunting** (`costophrenic_angle_blunting`)
- **Pleural effusion** (`pleural_effusion`)
  - **Hydropneumothorax** (`hydropneumothorax`)
  - **Loculated/fissural effusion** (`loculated_or_fissural_pleural_effusion`)
- **Pleural nodule or mass** (`pleural_nodule_or_mass`)
- **Pleural thickening** (`pleural_thickening`)
  - **Pleural calcification** (`pleural_calcification`) — parents: Calcification
  - **Pleural plaque** (`pleural_plaque`)
- **Pneumothorax** (`pneumothorax`)
  - **Hydropneumothorax** (`hydropneumothorax`)

### Cardiomediastinal

Cardiac, vascular, hilar, and mediastinal contour or content findings.

- **Aortic calcification** (`aortic_calcification`) — parents: Calcification
- **Aortic tortuosity/elongation** (`aortic_tortuosity_or_elongation`)
- **Cardiac/valvular/pericardial calcification** (`cardiac_or_valvular_calcification`) — parents: Calcification
- **Enlarged cardiomediastinal silhouette** (`enlarged_cardiomediastinal_silhouette`)
  - **Aortic enlargement/ectasia** (`aortic_enlargement_or_ectasia`)
  - **Enlarged cardiac silhouette** (`enlarged_cardiac_silhouette`)
  - **Mediastinal widening** (`mediastinal_widening`)
- **Enlarged pulmonary artery** (`enlarged_pulmonary_artery`)
- **Hilar enlargement** (`hilar_enlargement`)
- **Hilar or mediastinal nodal enlargement** (`hilar_or_mediastinal_nodal_enlargement`)
- **Mediastinal mass** (`mediastinal_mass`)
- **Mediastinal shift** (`mediastinal_shift`)
- **Pneumomediastinum** (`pneumomediastinum`)
- **Pulmonary vascular congestion** (`pulmonary_vascular_congestion`)

### Airway

Tracheal and bronchial findings.

- **Bronchial wall thickening** (`bronchial_wall_thickening`)
- **Bronchiectasis** (`bronchiectasis`)
- **Tracheal deviation** (`tracheal_deviation`)

### Diaphragm

Diaphragmatic contour, position, and hernia findings.

- **Elevated/eventrated hemidiaphragm** (`elevated_or_eventrated_hemidiaphragm`)
- **Flattened diaphragm** (`flattened_diaphragm`)
- **Hernia** (`hernia`)
  - **Hiatal hernia** (`hiatal_hernia`)

### Upper abdomen

Clinically important upper-abdominal findings visible on CXR.

- **Pneumoperitoneum** (`pneumoperitoneum`)

### Bone fracture findings

Organizational group for fractures of osseous structures visible on chest radiographs, modeled under the Bone fracture umbrella class.

- **Bone fracture** (`bone_fracture`)
  - **Clavicle fracture** (`clavicle_fracture`)
  - **Rib fracture** (`rib_fracture`)
  - **Vertebral compression fracture** (`vertebral_compression_fracture`)

### Musculoskeletal

Non-fracture thoracic osseous and spinal findings.

- **Degenerative osseous change** (`degenerative_osseous_change`)
- **Focal osseous lesion** (`focal_osseous_lesion`)
- **Kyphosis** (`kyphosis`)
- **Radiographic osteopenia** (`radiographic_osteopenia`)
- **Scoliosis** (`scoliosis`)

### Support device

Therapeutic lines, tubes, implants, prostheses, surgical hardware, and device-associated attributes modeled under the Support device present umbrella.

- **Support device present** (`support_device_present`)
  - **Breast implant** (`breast_implant`)
  - **Device abnormality** (`device_abnormality`)
  - **Pacemaker/ICD** (`pacemaker_or_icd`)
  - **Prosthetic cardiac valve** (`prosthetic_cardiac_valve`)
  - **Sternotomy wires** (`sternotomy_wires`)
  - **Support line present** (`support_line_present`)
    - **Central venous catheter** (`central_venous_catheter`)
  - **Support tube present** (`support_tube_present`)
    - **Chest tube** (`chest_tube`)
    - **Endotracheal tube** (`endotracheal_tube`)
    - **Enteric tube** (`enteric_tube`)
  - **Vascular graft/stent** (`vascular_graft_or_stent`)

### Soft tissue

Extra-osseous chest-wall and subcutaneous findings.

- **Subcutaneous emphysema** (`subcutaneous_emphysema`)

### Other

Image-recognizable findings not fitting another anatomic category.

- **Nontherapeutic foreign body** (`nontherapeutic_foreign_body`)

### Anatomic variant

Visible benign anatomic variants.

- **Azygos lobe** (`azygos_lobe`)

## Definition coverage

79/79 ontology classes have `.txt` definitions under `medvqa/prompts/cxr_classes/` (basename = class id).

## Multi-parent classes

- **Calcified pulmonary nodule or granuloma** (`calcified_pulmonary_nodule_or_granuloma`) ← Calcification, Pulmonary nodule or mass
- **Hydropneumothorax** (`hydropneumothorax`) ← Pleural effusion, Pneumothorax
- **Pleural calcification** (`pleural_calcification`) ← Pleural thickening, Calcification
