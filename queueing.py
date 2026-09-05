"""
queueing.py
------------
A small bounded worker pool that /recover submits work to, instead of
running the ML + decision + audit-log pipeline directly on the
request-handling thread.

This deliberately is NOT a Redis/RQ or Celery setup — for a project
this size, standing up and operating an external broker is infra cost
with no real benefit. What this *does* demonstrate is the pattern that
actually matters for webhook-shaped traffic: the request handler
enqueues work and waits on a bounded pool rather than doing unbounded
synchronous work per request. A burst of failed-payment webhooks now
queues up and drains at a controlled rate (MAX_WORKERS at a time)
instead of spawning unbounded concurrent model inference + DB writes,
and a pool that's genuinely saturated fails fast with backpressure
(HTTP 429) instead of piling up and falling over.

Swapping this for Redis/RQ or Celery in production is a matter of
changing submit()'s implementation — call sites (main.py) don't need
to change, since they already just do:

    result = queueing.submit(some_function, *args)
"""
import queue as _queue
from concurrent.futures import ThreadPoolExecutor

MAX_WORKERS = 4
MAX_QUEUE_DEPTH = 100

_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="recovery-worker")
_inflight = _queue.Queue(maxsize=MAX_QUEUE_DEPTH)


def submit(fn, *args, timeout: float = 15.0, **kwargs):
    """
    Enqueues fn(*args, **kwargs) on the worker pool and blocks for the
    result.

    Raises:
        queue.Full: the pool already has MAX_QUEUE_DEPTH requests
            in flight — this is the backpressure signal; the caller
            should turn it into an HTTP 429.
        TimeoutError: a worker didn't finish within `timeout` seconds.
    """
    try:
        _inflight.put_nowait(1)
    except _queue.Full:
        raise _queue.Full("recovery worker pool is at capacity, try again shortly")

    try:
        future = _executor.submit(fn, *args, **kwargs)
        return future.result(timeout=timeout)
    finally:
        _inflight.get_nowait()
