import itertools
import os
import re
from multiprocessing import Pool

# ================================ Pattern Loading ================================

_CXR_CLASSES_DIR = os.path.join(os.path.dirname(__file__), 'cxr_classes')
# Cache key: (directory, stem) so workdir copies do not collide with production.
_PATTERN_CACHE: dict[tuple[str, str], re.Pattern] = {}


def clear_pattern_cache() -> None:
    _PATTERN_CACHE.clear()


def load_pattern(name: str, directory: str | None = None) -> re.Pattern:
    """Load a VERBOSE regex pattern from ``<directory>/<name>.txt``.

    ``directory`` defaults to the package ``cxr_classes`` folder. Call
    :func:`clear_pattern_cache` after editing files on disk.
    """
    directory = os.path.abspath(directory or _CXR_CLASSES_DIR)
    key = (directory, name)
    if key not in _PATTERN_CACHE:
        path = os.path.join(directory, f'{name}.txt')
        with open(path, 'r', encoding='utf-8') as f:
            body = f.read()
        _PATTERN_CACHE[key] = re.compile(body, re.IGNORECASE | re.VERBOSE)
    return _PATTERN_CACHE[key]


def _load_pattern(name: str) -> re.Pattern:
    """Load from the package cxr_classes dir (used by composition tables)."""
    return load_pattern(name, directory=_CXR_CLASSES_DIR)


# Canonical ontology IDs as keys. Composition OR-includes YAML children
# (and optional non-ontology helper stems) for high-recall candidate finding.
# Label inheritance for released labels remains in canonical_cxr_ontology.yaml.
_CLASS_NAME_TO_REGEX_PATTERNS = {
    'calcification': [
        _load_pattern('calcification'),
        'calcified_pulmonary_nodule_or_granuloma',
        'pleural_calcification',
        'aortic_calcification',
        'cardiac_or_valvular_calcification',
    ],
    'lung_opacity': [
        _load_pattern('lung_opacity'),
        'ground_glass_opacity_pattern',
        'consolidation',
        'pneumonia',
        'atelectasis',
        'reticular_interstitial_opacity_pattern',
        'kerley_lines',
        'pulmonary_edema',
        'fibrotic_lung_change',
        'focal_pulmonary_scarring',
        'pulmonary_nodule_or_mass',
        # B1: infiltration / alveolar_pattern / air_bronchogram / abscess /
        # miliary_opacities helpers merged into lung_opacity.txt residual arms.
    ],
    'ground_glass_opacity_pattern': _load_pattern('ground_glass_opacity_pattern'),
    'consolidation': _load_pattern('consolidation'),
    'pneumonia': _load_pattern('pneumonia'),
    'tuberculosis': _load_pattern('tuberculosis'),
    'interstitial_lung_disease': _load_pattern('interstitial_lung_disease'),
    'atelectasis': [
        _load_pattern('atelectasis'),
        'lobar_atelectasis',
        'rounded_atelectasis',
    ],
    'lobar_atelectasis': _load_pattern('lobar_atelectasis'),
    'rounded_atelectasis': _load_pattern('rounded_atelectasis'),
    'reticular_interstitial_opacity_pattern': _load_pattern('reticular_interstitial_opacity_pattern'),
    'kerley_lines': _load_pattern('kerley_lines'),
    'pulmonary_edema': _load_pattern('pulmonary_edema'),
    'fibrotic_lung_change': _load_pattern('fibrotic_lung_change'),
    'focal_pulmonary_scarring': _load_pattern('focal_pulmonary_scarring'),
    'pulmonary_nodule_or_mass': [
        _load_pattern('pulmonary_nodule_or_mass'),
        'calcified_pulmonary_nodule_or_granuloma',
    ],
    'calcified_pulmonary_nodule_or_granuloma': _load_pattern('calcified_pulmonary_nodule_or_granuloma'),
    'pulmonary_cavity': _load_pattern('pulmonary_cavity'),
    'pulmonary_cyst_or_bulla': _load_pattern('pulmonary_cyst_or_bulla'),
    'emphysema': _load_pattern('emphysema'),
    'bronchiectasis': _load_pattern('bronchiectasis'),
    'bronchial_wall_thickening': _load_pattern('bronchial_wall_thickening'),
    'low_lung_volume': _load_pattern('low_lung_volume'),
    'hyperinflation': _load_pattern('hyperinflation'),
    'pleural_effusion': [
        _load_pattern('pleural_effusion'),
        'loculated_or_fissural_pleural_effusion',
        'hydropneumothorax',
    ],
    'loculated_or_fissural_pleural_effusion': _load_pattern('loculated_or_fissural_pleural_effusion'),
    'costophrenic_angle_blunting': _load_pattern('costophrenic_angle_blunting'),
    'pneumothorax': [
        _load_pattern('pneumothorax'),
        'hydropneumothorax',
    ],
    'hydropneumothorax': _load_pattern('hydropneumothorax'),
    'pleural_thickening': [
        _load_pattern('pleural_thickening'),
        'pleural_plaque',
        'pleural_calcification',
    ],
    'pleural_plaque': _load_pattern('pleural_plaque'),
    'pleural_calcification': _load_pattern('pleural_calcification'),
    'pleural_nodule_or_mass': _load_pattern('pleural_nodule_or_mass'),
    'enlarged_cardiomediastinal_silhouette': [
        _load_pattern('enlarged_cardiomediastinal_silhouette'),
        'enlarged_cardiac_silhouette',
        'aortic_enlargement_or_ectasia',
        'mediastinal_widening',
    ],
    'enlarged_cardiac_silhouette': _load_pattern('enlarged_cardiac_silhouette'),
    'pulmonary_vascular_congestion': _load_pattern('pulmonary_vascular_congestion'),
    'enlarged_pulmonary_artery': _load_pattern('enlarged_pulmonary_artery'),
    'aortic_enlargement_or_ectasia': _load_pattern('aortic_enlargement_or_ectasia'),
    'aortic_tortuosity_or_elongation': _load_pattern('aortic_tortuosity_or_elongation'),
    'aortic_calcification': _load_pattern('aortic_calcification'),
    'cardiac_or_valvular_calcification': _load_pattern('cardiac_or_valvular_calcification'),
    'hilar_enlargement': _load_pattern('hilar_enlargement'),
    'hilar_or_mediastinal_nodal_enlargement': _load_pattern('hilar_or_mediastinal_nodal_enlargement'),
    'mediastinal_widening': _load_pattern('mediastinal_widening'),
    'mediastinal_mass': _load_pattern('mediastinal_mass'),
    'mediastinal_shift': _load_pattern('mediastinal_shift'),
    'pneumomediastinum': _load_pattern('pneumomediastinum'),
    'tracheal_deviation': _load_pattern('tracheal_deviation'),
    'elevated_or_eventrated_hemidiaphragm': _load_pattern('elevated_or_eventrated_hemidiaphragm'),
    'flattened_diaphragm': _load_pattern('flattened_diaphragm'),
    'hernia': [
        _load_pattern('hernia'),
        'hiatal_hernia',
    ],
    'hiatal_hernia': _load_pattern('hiatal_hernia'),
    'pneumoperitoneum': _load_pattern('pneumoperitoneum'),
    'bone_fracture': [
        _load_pattern('bone_fracture'),
        'clavicle_fracture',
        'rib_fracture',
        'vertebral_compression_fracture',
    ],
    'clavicle_fracture': _load_pattern('clavicle_fracture'),
    'rib_fracture': _load_pattern('rib_fracture'),
    'vertebral_compression_fracture': _load_pattern('vertebral_compression_fracture'),
    'focal_osseous_lesion': _load_pattern('focal_osseous_lesion'),
    'degenerative_osseous_change': _load_pattern('degenerative_osseous_change'),
    'radiographic_osteopenia': _load_pattern('radiographic_osteopenia'),
    'scoliosis': _load_pattern('scoliosis'),
    'kyphosis': _load_pattern('kyphosis'),
    'support_device_present': [
        _load_pattern('support_device_present'),
        'support_line_present',
        'support_tube_present',
        'pacemaker_or_icd',
        'prosthetic_cardiac_valve',
        'vascular_graft_or_stent',
        'breast_implant',
        'sternotomy_wires',
        'device_abnormality',
    ],
    'support_line_present': [
        _load_pattern('support_line_present'),
        'central_venous_catheter',
    ],
    'support_tube_present': [
        _load_pattern('support_tube_present'),
        'endotracheal_tube',
        'enteric_tube',
        'chest_tube',
    ],
    'endotracheal_tube': _load_pattern('endotracheal_tube'),
    'enteric_tube': _load_pattern('enteric_tube'),
    'central_venous_catheter': _load_pattern('central_venous_catheter'),
    'chest_tube': _load_pattern('chest_tube'),
    'pacemaker_or_icd': _load_pattern('pacemaker_or_icd'),
    'prosthetic_cardiac_valve': _load_pattern('prosthetic_cardiac_valve'),
    'vascular_graft_or_stent': _load_pattern('vascular_graft_or_stent'),
    'breast_implant': _load_pattern('breast_implant'),
    'sternotomy_wires': _load_pattern('sternotomy_wires'),
    'device_abnormality': _load_pattern('device_abnormality'),
    'subcutaneous_emphysema': _load_pattern('subcutaneous_emphysema'),
    'nontherapeutic_foreign_body': _load_pattern('nontherapeutic_foreign_body'),
    'azygos_lobe': _load_pattern('azygos_lobe'),
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

PatternDefinition = re.Pattern | list[re.Pattern | str]

def _search_worker(pattern, text):
    """A simple worker to be used by multiprocessing pools."""
    return 1 if pattern.search(text) else 0

def collect_reports_matching_class(
    reports: list[str],
    class_name: str,
    class_match_cache: dict[str, list[int]],
    class_to_regex_patterns: dict[str, PatternDefinition] | None = _CLASS_NAME_TO_REGEX_PATTERNS,
    num_processes: int | None = 1,
) -> list[int]:
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
                raise TypeError(f"Expected sub_pattern to be a regex pattern or a string, got {type(sub_pattern)}")
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

    sorted_indices = sorted(matching_report_indices)
    class_match_cache[class_name] = sorted_indices
    return sorted_indices
