from multiprocessing import Pool
from functools import partial


def parmap(func, in_vals, args=[], kwargs={}, n_jobs=1):
    assert isinstance(n_jobs, int)
    assert n_jobs >= -1
    new_func = partial(func, *args, **kwargs)
    if n_jobs == 1:
        mapped = map(new_func, in_vals)
    else:
        if n_jobs == -1:
            pool = Pool()
        else:
            pool = Pool(processes=n_jobs)
        mapped = pool.map(new_func, in_vals)
        pool.close()
    return mapped
