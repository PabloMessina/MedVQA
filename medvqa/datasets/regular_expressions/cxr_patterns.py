import itertools
import os
import re
from multiprocessing import Pool
from typing import Dict, List, Optional, Set, Union

# ================================ Pattern Loading ================================

_CXR_CLASSES_DIR = os.path.join(os.path.dirname(__file__), 'cxr_classes')
_PATTERN_CACHE: Dict[str, re.Pattern] = {}


def _load_pattern(name: str) -> re.Pattern:
    """Load a VERBOSE regex pattern from cxr_classes/<name>.txt."""
    if name not in _PATTERN_CACHE:
        path = os.path.join(_CXR_CLASSES_DIR, f'{name}.txt')
        with open(path, 'r') as f:
            body = f.read()
        _PATTERN_CACHE[name] = re.compile(body, re.IGNORECASE | re.VERBOSE)
    return _PATTERN_CACHE[name]


# ================================ Class Name to Regex Mapping ================================

_CLASS_NAME_TO_REGEX_PATTERNS = {
    'Abscess': _load_pattern('abscess'),
    'Adenopathy': _load_pattern('adenopathy'),
    'Air Bronchogram': _load_pattern('air_bronchogram'),
    'Alveolar Pattern': _load_pattern('alveolar_pattern'),
    'Aortic Aneurysm': _load_pattern('aortic_aneurysm'),
    'Aortic Calcification': _load_pattern('aortic_calcification'),
    'Aortic Ectasia': _load_pattern('aortic_ectasia'),
    'Aortic Endoprosthesis': _load_pattern('aortic_endoprosthesis'),
    'Aortic Enlargement': [
        _load_pattern('aortic_enlargement'),
        'Aortic Aneurysm',
        'Aortic Ectasia',
        'Aortic Knob Enlargement',
        'Aortic Tortuosity',
    ],
    'Aortic Knob Enlargement': _load_pattern('aortic_knob_enlargement'),
    'Aortic Tortuosity': _load_pattern('aortic_tortuosity'),
    'Apical Cap': _load_pattern('apical_cap'),
    'Artificial Heart Valve': [
        _load_pattern('artificial_heart_valve'),
        'Artificial Aortic Heart Valve',
        'Artificial Mitral Heart Valve',
    ],
    'Artificial Aortic Heart Valve': _load_pattern('artificial_aortic_heart_valve'),
    'Artificial Mitral Heart Valve': _load_pattern('artificial_mitral_heart_valve'),
    'Atelectasis': [
        _load_pattern('atelectasis'),
        'Lobar Atelectasis',
        'Rounded Atelectasis',
    ],
    'Azygos Lobe': _load_pattern('azygos_lobe'),
    'Bulla': _load_pattern('bulla'),
    'Calcification': [
        _load_pattern('calcification'),
        'Aortic Calcification',
        'Pleural Calcification',
    ],
    'Callus Rib Fracture': _load_pattern('callus_rib_fracture'),
    'Cardiomegaly': _load_pattern('cardiomegaly'),
    'Catheter': [
        _load_pattern('catheter'),
        'Central Venous Catheter',
    ],
    'Central Venous Catheter': [
        _load_pattern('central_venous_catheter'),
        'Central Venous Catheter via Jugular Vein',
        'Central Venous Catheter via Subclavian Vein',
        'Reservoir Central Venous Catheter',
    ],
    'Central Venous Catheter via Jugular Vein': _load_pattern('central_venous_catheter_via_jugular_vein'),
    'Central Venous Catheter via Subclavian Vein': _load_pattern('central_venous_catheter_via_subclavian_vein'),
    'Chest Drain Tube': _load_pattern('chest_drain_tube'),
    'Clavicle Fracture': _load_pattern('clavicle_fracture'),
    'Consolidation': _load_pattern('consolidation'),
    'COPD/Emphysema': _load_pattern('copd_emphysema'),
    'Costophrenic Angle Blunting': _load_pattern('costophrenic_angle_blunting'),
    'Dual Chamber Device': _load_pattern('dual_chamber_device'),
    'Edema': _load_pattern('edema'),
    'Empyema': _load_pattern('empyema'),
    'Endotracheal Tube': _load_pattern('endotracheal_tube'),
    'Enlarged Cardiomediastinum': [
        'Cardiomegaly',
        'Mediastinal Enlargement',
    ],
    'Enlarged PA': _load_pattern('enlarged_pulmonary_artery'),
    'Pulmonary Fibrosis': _load_pattern('pulmonary_fibrosis'),
    'Fissural Effusion': [
        _load_pattern('fissural_effusion'),
        'Pseudotumor',
    ],
    'Fissural Abnormality': [
        'Fissural Effusion',
        'Fissural Thickening',
    ],
    'Fissural Thickening': [
        _load_pattern('fissure_thickening'),
        'Major Fissure Thickening',
        'Minor Fissure Thickening',
    ],
    'Fracture': [
        _load_pattern('fracture'),
        'Callus Rib Fracture',
        'Clavicle Fracture',
        'Humeral Fracture',
        'Rib Fracture',
        'Vertebral Fracture',
    ],
    'Gastrostomy Tube': _load_pattern('gastrostomy_tube'),
    'Granuloma': _load_pattern('granuloma'),
    'Ground Glass Pattern': _load_pattern('ground_glass_pattern'),
    'Hernia': [
        _load_pattern('hernia'),
        'Hiatal Hernia',
    ],
    'Hiatal Hernia': _load_pattern('hiatal_hernia'),
    'Hilar Enlargement': _load_pattern('hilar_enlargement'),
    'Humeral Fracture': _load_pattern('humeral_fracture'),
    'Humeral Prosthesis': _load_pattern('humeral_prosthesis'),
    'Hydropneumothorax': _load_pattern('hydropneumothorax'),
    'ICD': _load_pattern('icd'),
    'Infiltration': [
        _load_pattern('infiltration'),
        'Alveolar Pattern',
        'Consolidation',
    ],
    'Interstitial Lung Disease': [
        _load_pattern('interstitial_lung_disease'),
        'Reticulonodular Interstitial Pattern',
        'Reticular Interstitial Pattern',
        'Kerley Lines',
        'Ground Glass Pattern',
        'Miliary Opacities',
        'Pulmonary Fibrosis',
    ],
    'Kerley Lines': _load_pattern('kerley_lines'),
    'Kyphosis': _load_pattern('kyphosis'),
    'Lobar Atelectasis': _load_pattern('lobar_atelectasis'),
    'Loculated Pleural Effusion': _load_pattern('loculated_pleural_effusion'),
    'Lung Cavity': _load_pattern('lung_cavity'),
    'Lung Cyst': [
        _load_pattern('lung_cyst'),
        'Bulla',
    ],
    'Lung Lesion': [
        _load_pattern('lung_lesion'),
        'Granuloma',
        'Lung Tumor',
        'Nodule',
        'Pseudonodule',
        'Pulmonary Mass',
    ],
    'Lung Opacity': [
        _load_pattern('lung_opacity'),
        'Abscess',
        'Air Bronchogram',
        'Alveolar Pattern',
        'Atelectasis',
        'Consolidation',
        'Edema',
        'Granuloma',
        'Ground Glass Pattern',
        'Infiltration',
        'Kerley Lines',
        'Lung Cavity',
        'Lung Cyst',
        'Miliary Opacities',
        'Nodule',
        'Pseudonodule',
        'Pulmonary Fibrosis',
        'Pulmonary Mass',
        'Reticular Interstitial Pattern',
        'Reticulonodular Interstitial Pattern',
    ],
    'Lung Tumor': _load_pattern('lung_tumor'),
    'Major Fissure Thickening': _load_pattern('major_fissure_thickening'),
    'Mammary Prosthesis': _load_pattern('mammary_prosthesis'),
    'Mass': [
        _load_pattern('mass'),
        'Lung Tumor',
        'Pulmonary Mass',
        'Thoracic Masslike Finding',
    ],
    'Mediastinal Enlargement': _load_pattern('mediastinal_enlargement'),
    'Mediastinal Shift': _load_pattern('mediastinal_shift'),
    'Metal': _load_pattern('metal'),
    'Miliary Opacities': _load_pattern('miliary_opacities'),
    'Minor Fissure Thickening': _load_pattern('minor_fissure_thickening'),
    'Nasogastric Tube': _load_pattern('nasogastric_tube'),
    'Nodule': _load_pattern('nodule'),
    'Osteopenia': _load_pattern('osteopenia'),
    'Pacemaker': [
        _load_pattern('pacemaker'),
        'ICD',
        'Dual Chamber Device',
        'Single Chamber Device',
    ],
    'Pleural Calcification': [
        _load_pattern('pleural_calcification'),
        'Pleural Plaques',
    ],
    'Pleural Effusion': [
        _load_pattern('pleural_effusion'),
        'Costophrenic Angle Blunting',
        'Empyema',
        'Fissural Effusion',
        'Loculated Pleural Effusion',
    ],
    'Pleural Nodule/Mass': _load_pattern('pleural_nodule_mass'),
    'Pleural Plaques': _load_pattern('pleural_plaques'),
    'Pleural Scarring': _load_pattern('pleural_scarring'),
    'Pleural Thickening': [
        _load_pattern('pleural_thickening'),
        'Pleural Calcification',
        'Pleural Plaques',
        'Pleural Scarring',
        'Apical Cap',
    ],
    'Pneumomediastinum': _load_pattern('pneumomediastinum'),
    'Pneumonia': [
        _load_pattern('pneumonia'),
        'Consolidation',
        'Infiltration',
        'Lung Cavity',
        'Air Bronchogram',
    ],
    'Pneumoperitoneum': _load_pattern('pneumoperitoneum'),
    'Pneumothorax': [
        _load_pattern('pneumothorax'),
        'Hydropneumothorax',
    ],
    'Prosthesis': [
        _load_pattern('prosthesis'),
        'Artificial Heart Valve',
        'Humeral Prosthesis',
        'Mammary Prosthesis',
        'Aortic Endoprosthesis',
    ],
    'Pseudonodule': _load_pattern('pseudonodule'),
    'Pseudotumor': _load_pattern('pseudotumor'),
    'Pulmonary Embolism': _load_pattern('pulmonary_embolism'),
    'Pulmonary Hypertension': _load_pattern('pulmonary_hypertension'),
    'Pulmonary Infarction': _load_pattern('pulmonary_infarction'),
    'Pulmonary Mass': [
        _load_pattern('pulmonary_mass'),
        'Lung Tumor',
    ],
    'Pulmonary Vascular Congestion': _load_pattern('pulmonary_vascular_congestion'),
    'Reservoir Central Venous Catheter': _load_pattern('reservoir_central_venous_catheter'),
    'Reticular Interstitial Pattern': [
        _load_pattern('reticular_interstitial_pattern'),
        'Kerley Lines',
        'Pulmonary Fibrosis',
    ],
    'Reticulonodular Interstitial Pattern': [
        _load_pattern('reticulonodular_interstitial_pattern'),
        'Miliary Opacities',
    ],
    'Rib Fracture': _load_pattern('rib_fracture'),
    'Rounded Atelectasis': _load_pattern('rounded_atelectasis'),
    'Scoliosis': _load_pattern('scoliosis'),
    'Single Chamber Device': _load_pattern('single_chamber_device'),
    'Subcutaneous Emphysema': _load_pattern('subcutaneous_emphysema'),
    'Support Devices': [
        _load_pattern('support_devices'),
        'Pacemaker',
        'Catheter',
        'Endotracheal Tube',
        'Tracheostomy Tube',
        'Nasogastric Tube',
        'Chest Drain Tube',
        'Gastrostomy Tube',
        'Artificial Heart Valve',
        'Prosthesis',
        'Metal',
    ],
    'Thoracic Masslike Finding': _load_pattern('thoracic_masslike'),
    'Tracheostomy Tube': _load_pattern('tracheostomy_tube'),
    'Tuberculosis': _load_pattern('tuberculosis'),
    'Vertebral Fracture': _load_pattern('vertebral_fracture'),
}

def _sanity_check_regular_expressions():
    in_stack = set()
    visited = set()
    def dfs(class_name):
        assert isinstance(class_name, str), f"Expected class name to be a string, got {type(class_name)}"
        assert class_name in _CLASS_NAME_TO_REGEX_PATTERNS, f"Class name {class_name} not found in _CLASS_NAME_TO_REGEX_PATTERNS"
        assert class_name not in in_stack, f"Cycle detected: {class_name} -> {in_stack}"
        if class_name in visited:
            return # Already visited
        visited.add(class_name)
        in_stack.add(class_name)
        value = _CLASS_NAME_TO_REGEX_PATTERNS[class_name]
        if isinstance(value, list):
            items = value
            assert len(items) >= 2, f"Expected at least two items in a list for {class_name}"
            for item in items:
                assert isinstance(item, (str, re.Pattern)), f"Expected item to be a string or regular expression, got {type(item)}"
                if isinstance(item, str):
                    assert item in _CLASS_NAME_TO_REGEX_PATTERNS, f"Subclass {item} not found in _CLASS_NAME_TO_REGEX_PATTERNS"
                    dfs(item)
        else:
            assert isinstance(value, re.Pattern), f"Expected {class_name} to be a regular expression, got {type(value)}"
        in_stack.remove(class_name)
    for class_name in list(_CLASS_NAME_TO_REGEX_PATTERNS.keys()):
        dfs(class_name)

_sanity_check_regular_expressions() # If no error is raised, the regular expressions are valid


# ================================ Helper Functions ================================

PatternDefinition = Union[re.Pattern, List[Union[re.Pattern, str]]]

def _search_worker(pattern, text):
    """A simple worker to be used by multiprocessing pools."""
    return 1 if pattern.search(text) else 0

def collect_reports_matching_class(
    reports: List[str],
    class_name: str,
    class_match_cache: Dict[str, List[int]],
    class_to_regex_patterns: Optional[Dict[str, PatternDefinition]] = _CLASS_NAME_TO_REGEX_PATTERNS,
    num_processes: Optional[int] = 1,
) -> List[int]:
    """Recursively collects report indices that match regex patterns for a class.

    This function finds all report indices that match the regex pattern(s)
    associated with a given class name. It supports nested class definitions
    where a class can be defined by a list of regex patterns and other
    class names. Results are cached (memoized) to avoid re-computation for
    the same class.

    Args:
        reports: A list of report strings to search through.
        class_name: The name of the class to find matching reports for.
        class_match_cache: A dictionary used for memoization. It stores the
            list of matching report indices for each class name that has
            already been processed.
        class_to_regex_patterns: An optional dictionary mapping class names to
            their definitions. A definition can be a single compiled regex pattern
            or a list containing a mix of compiled regex patterns and other
            class names (strings). If not provided, the default dictionary
            _CLASS_NAME_TO_REGEX_PATTERNS will be used.
        num_processes: The number of processes to use for parallel processing.

    Returns:
        A sorted list of unique integer indices for the reports that match
        the specified class.

    Raises:
        ValueError: If the `class_name` is not found in the
            `class_to_regex_patterns` dictionary.
        TypeError: If an element within a pattern list is not a compiled
            regex pattern or a string.
    """
    # Check cache first to avoid re-computation
    if class_name in class_match_cache:
        return class_match_cache[class_name]

    if class_name not in _CLASS_NAME_TO_REGEX_PATTERNS:
        raise ValueError(
            f"Class '{class_name}' not found in the regex patterns."
        )

    pattern_definition = _CLASS_NAME_TO_REGEX_PATTERNS[class_name]
    matching_report_indices: set[int] = set()

    patterns_to_check = []
    if isinstance(pattern_definition, list):
        for sub_pattern in pattern_definition:
            if isinstance(sub_pattern, re.Pattern):
                patterns_to_check.append(sub_pattern)
            elif isinstance(sub_pattern, str):
                # This is a sub-class name, so recurse (this part is NOT parallelized)
                sub_matches = collect_reports_matching_class(
                    reports,
                    sub_pattern,
                    class_match_cache,
                    class_to_regex_patterns,
                    num_processes, # Pass the num_processes down
                )
                matching_report_indices.update(sub_matches)
            else:
                raise ValueError(f"Expected sub_pattern to be a regex pattern or a string, got {type(sub_pattern)}")
    else: # It's a single regex pattern
        patterns_to_check.append(pattern_definition)

    # Now, process all collected regex patterns
    for sub_pattern in patterns_to_check:
        if num_processes > 1:
            # Use a pool of workers to process reports in parallel
            with Pool(processes=num_processes) as pool:
                # Use starmap with the top-level worker function
                # We zip the pattern (repeated) with each report
                args = zip(itertools.repeat(sub_pattern), reports)
                results = pool.starmap(_search_worker, args)

            for i, match in enumerate(results):
                if match:
                    matching_report_indices.add(i)
        else:
            # Single-threaded processing
            for i, report_text in enumerate(reports):
                if sub_pattern.search(report_text):
                    matching_report_indices.add(i)

    sorted_indices = sorted(list(matching_report_indices))
    class_match_cache[class_name] = sorted_indices
    return sorted_indices
