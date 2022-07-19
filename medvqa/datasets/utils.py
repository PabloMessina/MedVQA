def deduplicate_indices(indices, report_ids):
    seen = set()
    indices_ = []
    for i in indices:
        if report_ids[i] not in seen:
            seen.add(report_ids[i])
            indices_.append(i)
    return indices_