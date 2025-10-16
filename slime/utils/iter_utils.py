from collections import defaultdict


# details: https://stackoverflow.com/questions/773/how-do-i-use-itertools-groupby
def group_by(iterable, key=None):
    """Similar to itertools.groupby, but do not require iterable to be sorted"""
    ret = defaultdict(list)
    for item in iterable:
        ret[key(item) if key is not None else item].append(item)
    return dict(ret)
