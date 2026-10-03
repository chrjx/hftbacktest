import os
import subprocess
import sys
import textwrap
import unittest

import numpy as np

from hftbacktest import DEPTH_EVENT, EXCH_EVENT, LOCAL_EVENT
from hftbacktest.data import correct_event_order, validate_event_order
from hftbacktest.types import event_dtype


def make_events(n: int, seed: int) -> np.ndarray:
    """Rows whose exchange timestamps are partly reversed relative to local timestamps."""
    rng = np.random.default_rng(seed)
    data = np.zeros(n, event_dtype)
    data['ev'] = DEPTH_EVENT
    data['local_ts'] = np.sort(rng.integers(1_000, 1_000 + 10 * n, n))
    data['exch_ts'] = data['local_ts'] - rng.integers(1, 50, n)
    data['px'] = rng.integers(1, 1000, n).astype(np.float64)
    data['qty'] = rng.integers(1, 100, n).astype(np.float64)
    return data


def reference_event_order(data, sorted_exch_index, sorted_local_index):
    """Plain-Python version of the documented algorithm."""
    n = len(data)
    out = []
    i = j = 0
    while i < n or j < n:
        if i < n and j < n:
            e = data[sorted_exch_index[i]]
            l = data[sorted_local_index[j]]
            if e['exch_ts'] == l['exch_ts'] and e['local_ts'] == l['local_ts']:
                out.append((e, EXCH_EVENT | LOCAL_EVENT))
                i += 1
                j += 1
            elif e['exch_ts'] < l['exch_ts'] or (
                    e['exch_ts'] == l['exch_ts'] and e['local_ts'] < l['local_ts']):
                out.append((e, EXCH_EVENT))
                i += 1
            else:
                out.append((l, LOCAL_EVENT))
                j += 1
        elif i < n:
            out.append((data[sorted_exch_index[i]], EXCH_EVENT))
            i += 1
        else:
            out.append((data[sorted_local_index[j]], LOCAL_EVENT))
            j += 1
    result = np.zeros(len(out), event_dtype)
    for k, (row, flag) in enumerate(out):
        result[k] = row
        result[k]['ev'] = row['ev'] | flag
    return result


def sorted_indices(data):
    return (
        np.argsort(data['exch_ts'], kind='mergesort'),
        np.argsort(data['local_ts'], kind='mergesort'),
    )


class TestCorrectEventOrder(unittest.TestCase):
    def test_matches_reference(self):
        for n, seed in [(1, 0), (2, 1), (10, 2), (1_000, 3), (200_000, 4)]:
            data = make_events(n, seed)
            exch_idx, local_idx = sorted_indices(data)
            out = correct_event_order(data, exch_idx, local_idx)
            np.testing.assert_array_equal(out, reference_event_order(data, exch_idx, local_idx))
            validate_event_order(out)

    def test_exchange_stream_finishes_first(self):
        # The last row by local time has the earliest exchange time, so the exchange-sorted stream
        # is exhausted while local rows remain.
        data = np.zeros(3, event_dtype)
        data['ev'] = DEPTH_EVENT
        data['exch_ts'] = [10, 20, 5]
        data['local_ts'] = [30, 40, 50]
        exch_idx, local_idx = sorted_indices(data)
        out = correct_event_order(data, exch_idx, local_idx)
        np.testing.assert_array_equal(out, reference_event_order(data, exch_idx, local_idx))
        validate_event_order(out)

    def test_no_out_of_bounds_reads(self):
        # Numba skips bounds checks by default, so an out-of-bounds read only shows up as garbage
        # or a crash. Run with NUMBA_BOUNDSCHECK=1 in a fresh interpreter to turn it into an IndexError.
        script = textwrap.dedent('''
            import numpy as np
            from hftbacktest import DEPTH_EVENT
            from hftbacktest.data import correct_event_order
            from hftbacktest.types import event_dtype

            data = np.zeros(3, event_dtype)
            data['ev'] = DEPTH_EVENT
            data['exch_ts'] = [10, 20, 5]
            data['local_ts'] = [30, 40, 50]
            correct_event_order(
                data,
                np.argsort(data['exch_ts'], kind='mergesort'),
                np.argsort(data['local_ts'], kind='mergesort'),
            )
        ''')
        env = dict(os.environ, NUMBA_BOUNDSCHECK='1', NUMBA_CACHE_DIR='')
        proc = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == '__main__':
    unittest.main()
