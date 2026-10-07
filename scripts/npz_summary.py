#!/usr/bin/env python3
import numpy as np
import sys
path='logs/evaluations.npz'
try:
    data=np.load(path, allow_pickle=True)
except Exception as e:
    print('ERROR loading', path, e, file=sys.stderr)
    sys.exit(2)
keys=list(data.keys())
print('FILE:', path)
print('KEYS:', keys)
for k in keys:
    v=data[k]
    try:
        arr=np.array(v)
        print('\n---', k)
        print('shape:', getattr(arr, 'shape', None))
        print('dtype:', getattr(arr, 'dtype', None))
        size = arr.size if hasattr(arr, 'size') else None
        print('size:', size)
        if size and size > 0:
            if np.issubdtype(arr.dtype, np.number):
                # Use nan-safe stats
                mn = float(np.nanmin(arr))
                mx = float(np.nanmax(arr))
                mean = float(np.nanmean(arr))
                print('min:', mn, 'max:', mx, 'mean:', mean)
            else:
                flat = arr.flatten()
                samples = [repr(x) for x in flat[:5]]
                print('samples:', samples)
    except Exception as e:
        print('ERROR for key', k, e)
print('\nDone.')
