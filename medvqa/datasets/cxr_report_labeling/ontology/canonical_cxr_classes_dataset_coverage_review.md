# Dataset-to-ontology coverage sanity check

Reverse audit of `canonical_cxr_ontology.yaml` (ontology version 1.0.0): every source label is checked against the current ontology.

**Coverage terminology**

- **Yes**: an exact, near-exact, or safe parent/fallback class exists (including when the source label is narrower than an ontology umbrella).
- **Partial**: the ontology captures only related morphology, a visual surrogate, or a broader class that cannot safely receive the source label without additional evidence.
- **No**: there is no usable ontology class (often deliberate for etiologic diagnoses CXR cannot establish).

## CXR-LT 2024

| # | Source class | Captured? | Ontology class(es) | Notes |
|---:|---|---|---|---|
| 1 | Adenopathy | **Yes** | `Hilar or mediastinal nodal enlargement` (near); `Hilar enlargement` (partial); `Mediastinal mass` (partial) |  |
| 2 | Atelectasis | **Yes** | `Atelectasis` |  |
| 3 | Azygos Lobe | **Yes** | `Azygos lobe` |  |
| 4 | Calcification of the Aorta | **Yes** | `Aortic calcification`; `Calcification` (narrower) |  |
| 5 | Cardiomegaly | **Yes** | `Enlarged cardiac silhouette` (near) |  |
| 6 | Clavicle Fracture | **Yes** | `Clavicle fracture` |  |
| 7 | Consolidation | **Yes** | `Consolidation` |  |
| 8 | Edema | **Yes** | `Pulmonary edema` (near); `Kerley lines` (partial); `Pulmonary vascular congestion` (partial) |  |
| 9 | Emphysema | **Yes** | `Emphysema`; `Flattened diaphragm` (partial); `Hyperinflation` (partial) |  |
| 10 | Enlarged Cardiomediastinum | **Yes** | `Enlarged cardiomediastinal silhouette`; `Enlarged cardiac silhouette` (broader); `Mediastinal widening` (broader); `Aortic enlargement/ectasia` (partial) |  |
| 11 | Fibrosis | **Yes** | `Fibrotic lung change` (near); `Focal pulmonary scarring` (broader); `Interstitial lung disease` (narrower); `Reticular interstitial opacity/pattern` (partial) |  |
| 12 | Fissure | **Partial** | `Loculated/fissural effusion` (partial) | Surrogate / broader / incomplete match; do not treat as equivalent. |
| 13 | Fracture | **Yes** | `Bone fracture`; `Vertebral compression fracture` (broader); `Focal osseous lesion` (partial) |  |
| 14 | Granuloma | **Yes** | `Calcified pulmonary nodule or granuloma` (near) |  |
| 15 | Hernia | **Yes** | `Hernia`; `Hiatal hernia` (broader) |  |
| 16 | Hydropneumothorax | **Yes** | `Hydropneumothorax` |  |
| 17 | Infarction | **No** | — | Deliberate omission or no coherent image target. |
| 18 | Infiltration | **Partial** | `Ground-glass opacity/pattern` (partial); `Lung opacity` (partial); `Reticular interstitial opacity/pattern` (partial) | Surrogate / broader / incomplete match; do not treat as equivalent. |
| 19 | Kyphosis | **Yes** | `Kyphosis` |  |
| 20 | Lobar Atelectasis | **Yes** | `Lobar atelectasis` |  |
| 21 | Lung Lesion | **Yes** | `Pulmonary nodule or mass` (near) |  |
| 22 | Lung Opacity | **Yes** | `Lung opacity`; `Ground-glass opacity/pattern` (broader) |  |
| 23 | Mass | **Yes** | `Pulmonary nodule or mass`; `Mediastinal mass` (partial); `Pleural nodule or mass` (partial) |  |
| 24 | Nodule | **Yes** | `Pulmonary nodule or mass` |  |
| 25 | Normal | **Yes** | `No finding` (near) |  |
| 26 | Pleural Effusion | **Yes** | `Pleural effusion`; `Loculated/fissural effusion` (broader); `Costophrenic angle blunting` (partial) |  |
| 27 | Pleural Other | **Partial** | `Pleural calcification` (broader); `Pleural nodule or mass` (broader); `Pleural plaque` (broader); `Costophrenic angle blunting` (partial) | Surrogate / broader / incomplete match; do not treat as equivalent. |
| 28 | Pleural Thickening | **Yes** | `Pleural thickening` |  |
| 29 | Pneumomediastinum | **Yes** | `Pneumomediastinum` |  |
| 30 | Pneumonia | **Yes** | `Pneumonia` |  |
| 31 | Pneumoperitoneum | **Yes** | `Pneumoperitoneum` |  |
| 32 | Pneumothorax | **Yes** | `Pneumothorax` |  |
| 33 | Pulmonary Embolism | **No** | — | Deliberate omission or no coherent image target. |
| 34 | Pulmonary Hypertension | **Partial** | `Enlarged pulmonary artery` (partial) | Surrogate / broader / incomplete match; do not treat as equivalent. |
| 35 | Rib Fracture | **Yes** | `Rib fracture` |  |
| 36 | Round(ed) Atelectasis | **Yes** | `Rounded atelectasis` |  |
| 37 | Subcutaneous Emphysema | **Yes** | `Subcutaneous emphysema` |  |
| 38 | Support Devices | **Yes** | `Support device present`; `Central venous catheter` (broader); `Chest tube` (broader); `Endotracheal tube` (broader); `Enteric tube` (broader); `Pacemaker/ICD` (broader); `Prosthetic cardiac valve` (broader); `Sternotomy wires` (broader); `Support line present` (broader); `Support tube present` (broader); `Vascular graft/stent` (broader); `Device abnormality` (partial) |  |
| 39 | Tortuous Aorta | **Yes** | `Aortic tortuosity/elongation` |  |
| 40 | Tuberculosis | **Yes** | `Tuberculosis` (near); `Pulmonary cavity` (partial) |  |
| 41 | Bulla | **Yes** | `Pulmonary cyst or bulla` |  |
| 42 | Cardiomyopathy | **No** | — | Deliberate omission or no coherent image target. |
| 43 | Hilum | **Partial** | `Enlarged pulmonary artery` (partial); `Hilar enlargement` (partial) | Surrogate / broader / incomplete match; do not treat as equivalent. |
| 44 | Osteopenia | **Yes** | `Radiographic osteopenia` (near) |  |
| 45 | Scoliosis | **Yes** | `Scoliosis` |  |

Summary: Yes=37, Partial=5, No=3 (n=45).

## PUC-CXR

| # | Source class | Captured? | Ontology class(es) | Notes |
|---:|---|---|---|---|
| 1 | Abnormal foreign body | **Yes** | `Nontherapeutic foreign body` (near) |  |
| 2 | Aortic atheromatosis | **Partial** | `Aortic calcification` (partial) | Surrogate / broader / incomplete match; do not treat as equivalent. |
| 3 | Aortic elongation | **Yes** | `Aortic tortuosity/elongation`; `Aortic enlargement/ectasia` (partial) |  |
| 4 | Aortic endoprosthesis | **Yes** | `Vascular graft/stent` (narrower) |  |
| 5 | Artificial heart valve | **Yes** | `Prosthetic cardiac valve` (near); `Support device present` (narrower) |  |
| 6 | Atelectasis | **Yes** | `Atelectasis`; `Lobar atelectasis` (broader); `Rounded atelectasis` (broader) |  |
| 7 | Bone lesion | **Yes** | `Focal osseous lesion` (near) |  |
| 8 | Bronchiectasis | **Yes** | `Bronchiectasis` |  |
| 9 | Bulla | **Yes** | `Pulmonary cyst or bulla` |  |
| 10 | Calcified granuloma | **Yes** | `Calcified pulmonary nodule or granuloma` (near); `Calcification` (narrower) |  |
| 11 | Calcified heart valve | **Yes** | `Calcification` (narrower); `Cardiac/valvular/pericardial calcification` (narrower) |  |
| 12 | Cardiomegaly | **Yes** | `Enlarged cardiac silhouette` (near); `Enlarged cardiomediastinal silhouette` (narrower) |  |
| 13 | Cavitation | **Yes** | `Pulmonary cavity` (near) |  |
| 14 | Consolidation | **Yes** | `Consolidation` |  |
| 15 | Costophrenic angle blunting | **Yes** | `Costophrenic angle blunting` |  |
| 16 | Degenerative changes | **Yes** | `Degenerative osseous change` (near) |  |
| 17 | Edema | **Yes** | `Pulmonary edema` (near) |  |
| 18 | Flattened diaphragm | **Yes** | `Flattened diaphragm`; `Hyperinflation` (partial) |  |
| 19 | Fracture | **Yes** | `Bone fracture`; `Clavicle fracture` (broader); `Vertebral compression fracture` (broader) |  |
| 20 | Granuloma | **Partial** | `Calcified pulmonary nodule or granuloma` (partial); `Pulmonary nodule or mass` (partial) | Surrogate / broader / incomplete match; do not treat as equivalent. |
| 21 | Ground glass pattern | **Yes** | `Ground-glass opacity/pattern`; `Lung opacity` (narrower) |  |
| 22 | Hemidiaphragm elevation | **Yes** | `Elevated/eventrated hemidiaphragm` |  |
| 23 | Hiatal hernia | **Yes** | `Hiatal hernia`; `Hernia` (narrower) |  |
| 24 | Hilar enlargement | **Yes** | `Hilar enlargement`; `Hilar or mediastinal nodal enlargement` (partial) |  |
| 25 | Hyperinflated lung | **Yes** | `Hyperinflation` (near) |  |
| 26 | Kerley lines | **Yes** | `Kerley lines`; `Pulmonary edema` (narrower) |  |
| 27 | Lines | **Yes** | `Support line present`; `Central venous catheter` (broader); `Support device present` (narrower); `Device abnormality` (partial) |  |
| 28 | Lung Mass | **Yes** | `Pulmonary nodule or mass` |  |
| 29 | Mediastinal enlargement | **Yes** | `Mediastinal widening` (near); `Enlarged cardiomediastinal silhouette` (narrower); `Hilar or mediastinal nodal enlargement` (partial) |  |
| 30 | Mediastinal mass | **Yes** | `Mediastinal mass` |  |
| 31 | Nodule | **Yes** | `Pulmonary nodule or mass` |  |
| 32 | Osteopenia | **Yes** | `Radiographic osteopenia` (near) |  |
| 33 | Pacemaker | **Yes** | `Pacemaker/ICD` (narrower); `Support device present` (narrower) |  |
| 34 | Pleural effusion | **Yes** | `Pleural effusion`; `Loculated/fissural effusion` (broader); `Hydropneumothorax` (partial) |  |
| 35 | Pleural mass | **Yes** | `Pleural nodule or mass` (near) |  |
| 36 | Pleural plaques | **Yes** | `Pleural plaque`; `Calcification` (narrower); `Pleural calcification` (partial) |  |
| 37 | Pleural thickening | **Yes** | `Pleural thickening` |  |
| 38 | Pneumonia | **Yes** | `Pneumonia` |  |
| 39 | Pneumothorax | **Yes** | `Pneumothorax`; `Hydropneumothorax` (partial) |  |
| 40 | Pulmonary fibrosis | **Yes** | `Fibrotic lung change` (near); `Focal pulmonary scarring` (broader); `Interstitial lung disease` (narrower) |  |
| 41 | Reticular interstitial pattern | **Yes** | `Reticular interstitial opacity/pattern`; `Interstitial lung disease` (partial) |  |
| 42 | Rib fracture | **Yes** | `Rib fracture` |  |
| 43 | Scoliosis | **Yes** | `Scoliosis` |  |
| 44 | Sternotomy | **Yes** | `Sternotomy wires` (near) |  |
| 45 | Subcutaneous emphysema | **Yes** | `Subcutaneous emphysema` |  |
| 46 | Tubes | **Yes** | `Support tube present`; `Chest tube` (broader); `Endotracheal tube` (broader); `Enteric tube` (broader); `Support device present` (narrower); `Device abnormality` (partial) |  |
| 47 | Vascular redistribution | **Yes** | `Pulmonary vascular congestion` (near) |  |

Summary: Yes=45, Partial=2, No=0 (n=47).

## VinDr-CXR

| # | Source class | Captured? | Ontology class(es) | Notes |
|---:|---|---|---|---|
| 1 | Aortic enlargement | **Yes** | `Aortic enlargement/ectasia`; `Enlarged cardiomediastinal silhouette` (narrower); `Aortic tortuosity/elongation` (partial) |  |
| 2 | Atelectasis | **Yes** | `Atelectasis`; `Lobar atelectasis` (broader); `Rounded atelectasis` (broader) |  |
| 3 | Calcification | **Yes** | `Calcification`; `Aortic calcification` (broader); `Calcified pulmonary nodule or granuloma` (broader); `Cardiac/valvular/pericardial calcification` (broader); `Pleural calcification` (broader); `Pleural plaque` (broader) |  |
| 4 | Cardiomegaly | **Yes** | `Enlarged cardiac silhouette` (near); `Enlarged cardiomediastinal silhouette` (narrower) |  |
| 5 | Clavicle fracture | **Yes** | `Clavicle fracture`; `Bone fracture` (narrower) |  |
| 6 | Consolidation | **Yes** | `Consolidation` |  |
| 7 | Edema | **Yes** | `Pulmonary edema` (near); `Kerley lines` (partial); `Pulmonary vascular congestion` (partial) |  |
| 8 | Emphysema | **Yes** | `Emphysema`; `Flattened diaphragm` (partial); `Hyperinflation` (partial); `Pulmonary cyst or bulla` (partial) |  |
| 9 | Enlarged PA | **Yes** | `Enlarged pulmonary artery` |  |
| 10 | ILD | **Yes** | `Interstitial lung disease`; `Fibrotic lung change` (partial); `Reticular interstitial opacity/pattern` (partial) |  |
| 11 | Infiltration | **Partial** | `Ground-glass opacity/pattern` (partial); `Lung opacity` (partial) | Surrogate / broader / incomplete match; do not treat as equivalent. |
| 12 | Lung Opacity | **Yes** | `Lung opacity`; `Ground-glass opacity/pattern` (broader) |  |
| 13 | Lung cavity | **Yes** | `Pulmonary cavity` |  |
| 14 | Lung cyst | **Yes** | `Pulmonary cyst or bulla` (near) |  |
| 15 | Mediastinal shift | **Yes** | `Mediastinal shift`; `Tracheal deviation` (partial) |  |
| 16 | Nodule/Mass | **Yes** | `Pulmonary nodule or mass` |  |
| 17 | Pleural effusion | **Yes** | `Pleural effusion`; `Loculated/fissural effusion` (broader); `Costophrenic angle blunting` (partial); `Hydropneumothorax` (partial) |  |
| 18 | Pleural thickening | **Yes** | `Pleural thickening`; `Pleural plaque` (partial) |  |
| 19 | Pneumothorax | **Yes** | `Pneumothorax`; `Hydropneumothorax` (partial) |  |
| 20 | Pulmonary fibrosis | **Yes** | `Fibrotic lung change` (near); `Focal pulmonary scarring` (broader); `Reticular interstitial opacity/pattern` (partial) |  |
| 21 | Rib fracture | **Yes** | `Rib fracture`; `Bone fracture` (narrower) |  |
| 22 | Other lesion | **Yes** | `Focal osseous lesion` (broader); `Mediastinal mass` (broader); `Nontherapeutic foreign body` (broader); `Pleural nodule or mass` (broader) |  |
| 23 | COPD | **Partial** | `Emphysema` (partial); `Flattened diaphragm` (partial); `Hyperinflation` (partial) | Surrogate / broader / incomplete match; do not treat as equivalent. |
| 24 | Lung tumor | **Partial** | `Pulmonary nodule or mass` (partial) | Surrogate / broader / incomplete match; do not treat as equivalent. |
| 25 | Pneumonia | **Yes** | `Pneumonia` |  |
| 26 | Tuberculosis | **Yes** | `Tuberculosis` (near); `Pulmonary cavity` (partial) |  |
| 27 | Other disease | **No** | — | Deliberate omission or no coherent image target. |
| 28 | No finding | **Yes** | `No finding` |  |

Summary: Yes=24, Partial=3, No=1 (n=28).

## Conclusion

Relative to earlier proposal reviews, the current ontology incorporates:

1. `Enlarged cardiomediastinal silhouette` as umbrella for undifferentiated enlargement
2. `Calcification` as umbrella for unlocalized calcific density
3. `Ground-glass opacity/pattern`, `Reticular interstitial opacity/pattern`, and `Kerley lines`
4. `Support line present` and `Support tube present` as line/tube umbrellas under `Support device present`
5. Merged `Pulmonary nodule or mass` and `Pulmonary cyst or bulla` (CXR size/wall distinctions often unreliable)
6. `Bone fracture` umbrella (replacing the prior thoracic-fracture naming) with site-specific children
7. `Device abnormality` for visible device position/integrity anomalies (replacing device-malposition-only framing)
8. Diagnostic labels retained nonexclusively: pneumonia, tuberculosis, ILD

Incomplete coverage of etiologic/clinical source labels (`Pulmonary Embolism`, `Infarction`, `COPD`, `Pulmonary Hypertension`, `Cardiomyopathy`, `Lung tumor`, `Other disease`) remains intentional.

Definition files: 79/79 ontology classes currently have `.txt` definitions under `medvqa/prompts/cxr_classes/`.
