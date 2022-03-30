def average_ignoring_nones(values):
    acc = 0
    count = 0
    for x in values:
        if x is not None:
            acc += x
            count += 1
    return acc / count if count > 0 else 0
