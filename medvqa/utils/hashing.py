_B = 1777771
_M = [999727999, 1070777777]
_MASK32 = (1 << 32) - 1

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
            h = (prev_hash[1] >> (32 * k)) & _MASK32
            for c in x:
                h = (h * _B + ord(c)) % _M[k]
            new_hash = (new_hash << 32) | h
        new_count = prev_hash[0] + len(x)
    else:
        assert type(x) is int
        for k in range(2):
            h = (prev_hash[1] >> (32 * k)) & _MASK32
            h = (h * _B + x) % _M[k]
            new_hash = (new_hash << 32) | h
        new_count = prev_hash[0] + 1
    return (new_count, new_hash)