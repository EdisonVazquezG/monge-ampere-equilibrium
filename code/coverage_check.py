"""Exact combinatorial audit of quadtree disk coverage and angular boundary coverage."""
from fractions import Fraction


def check_cover(leaves, discarded, arcs):
    cells = set()
    for row in list(leaves)+list(discarded):
        i, j, level = (int(row[k]) for k in ('ix', 'iy', 'level'))
        assert level >= 0 and 0 <= i < 2**level and 0 <= j < 2**level
        key = (i, j, level)
        assert key not in cells, 'Duplicate cell'
        cells.add(key)
    for i, j, level in cells:
        for ancestor in range(level):
            shift = level-ancestor
            assert (i >> shift, j >> shift, ancestor) not in cells, 'Overlapping cells'
    assert sum((Fraction(1, 4**level) for _, _, level in cells), Fraction(0)) == 1, 'Uncovered square area'
    for row in discarded:
        i, j, level = (int(row[k]) for k in ('ix', 'iy', 'level')); n = 2**level
        ds = []
        for k in (i, j):
            lo, hi = 2*k-n, 2*k+2-n
            ds.append(0 if lo <= 0 <= hi else min(abs(lo), abs(hi)))
        assert sum(d*d for d in ds) > n*n, 'Discarded box touches the disk'
    intervals = set()
    for row in arcs:
        index, level = int(row['index']), int(row['level'])
        assert level >= 0 and 0 <= index < 2**level
        assert (index, level) not in intervals
        intervals.add((index, level))
    for index, level in intervals:
        for ancestor in range(level):
            assert (index >> (level-ancestor), ancestor) not in intervals
    assert sum((Fraction(1, 2**level) for _, level in intervals), Fraction(0)) == 1, 'Boundary gap'
    return True
