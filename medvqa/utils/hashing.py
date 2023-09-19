_B = 1777771
_M = [999727999, 1070777777]
_MASK32 = (1 << 32) - 1
_MAXLEN = 1000000
_POW = [[None] * _MAXLEN, [None] * _MAXLEN] # _POW[k][i] = _B^i mod _M[k]
for k in range(2):
    _POW[k][0] = 1
    for i in range(1, _MAXLEN):
        _POW[k][i] = (_POW[k][i - 1] * _B) % _M[k]

def hash_string(s):
    hash = 0
    for k in range(2):
        h = 0
        for c in s:
            h = (h * _B + ord(c)) % _M[k]
        hash = (hash << 32) | h
    return (len(s), hash)

def update_hash(prev_hash, x):
    new_count = 0
    new_hash = 0
    if type(x) is str:
        for k in range(2):
            h = (prev_hash[1] >> (32 * (1 - k))) & _MASK32
            for c in x:
                h = (h * _B + ord(c)) % _M[k]
            new_hash = (new_hash << 32) | h
        new_count = prev_hash[0] + len(x)
    elif type(x) is int:
        for k in range(2):
            h = (prev_hash[1] >> (32 * (1 - k))) & _MASK32
            h = (h * _B + x) % _M[k]
            new_hash = (new_hash << 32) | h
        new_count = prev_hash[0] + 1
    else:
        assert type(x) is tuple
        assert len(x) == 2
        prev_count = prev_hash[0]
        prev_hash = prev_hash[1]
        x_count = x[0]
        x_hash = x[1]
        new_count = prev_count + x_count
        for k in range(2):
            h_prev = (prev_hash >> (32 * (1 - k))) & _MASK32
            h_x = (x_hash >> (32 * (1 - k))) & _MASK32
            h = (h_prev * _POW[k][x_count] + h_x) % _M[k]
            new_hash = (new_hash << 32) | h
    return (new_count, new_hash)

_shared_strings = None
def compute_hashes_in_parallel(strings, num_workers=None):
    global _shared_strings
    _shared_strings = strings
    import multiprocessing
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    else:
        num_workers = min(num_workers, multiprocessing.cpu_count())
    with multiprocessing.Pool(num_workers) as pool:
        hashes = pool.map(hash_string, _shared_strings)
    return hashes

def hash_string_list(l, in_parallel=True):
    assert type(l) is list
    assert len(l) > 0
    assert type(l[0]) is str
    if in_parallel:
        hashes = compute_hashes_in_parallel(l)
        hash = (0, 0)
        for h in hashes:
            hash = update_hash(hash, h)
    else:
        hash = (0, 0)
        for s in l:
            hash = update_hash(hash, s)
    return hash