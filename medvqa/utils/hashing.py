_B = 1777771
_M = [999727999, 1070777777]

def hash_string(s):
    hash = 0
    for k in range(2):
        h = 0
        for c in s:
            h = (h * _B + ord(c)) % _M[k]
        hash = (hash << 32) | h
    return (len(s), hash)
