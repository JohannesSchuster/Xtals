import time

def timed(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed_us = (end - start) * 1_000_000
        print(f"{func.__name__} took {elapsed_us:.2f} µs")
        return result
    return wrapper