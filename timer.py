import time
import functools

class Timer:
    def __init__(self):
        self._start_time = None
        self._end_time = None

    def start(self):
        self._start_time = time.perf_counter()
        self._end_time = None

    def stop(self):
        if self._start_time is None:
            raise RuntimeError("Timer was not started.")
        self._end_time = time.perf_counter()
        return self.elapsed

    @property
    def elapsed(self):
        if self._start_time is None:
            return 0.0
        if self._end_time is not None:
            return self._end_time - self._start_time
        return time.perf_counter() - self._start_time
    
def timed(fn):
    """Decorator to time a function and print its runtime."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[timed] {fn.__qualname__} took {elapsed:.4f} s")
        return result
    return wrapper