import httpx
from tenacity import wait_exponential, wait_fixed, wait_random

from resilient_httpx.retry import RetryPolicy


def test_defaults():
    policy = RetryPolicy()
    assert policy.max_attempts == 3
    assert policy.backoff == "exponential"
    assert policy.min_wait == 1.0
    assert policy.max_wait == 30.0
    assert policy.retry_on == [429, 500, 502, 503, 504]
    assert httpx.TimeoutException in policy.retry_on_exception
    assert httpx.ConnectError in policy.retry_on_exception


def test_build_retrying_exponential():
    retrying = RetryPolicy(backoff="exponential", min_wait=2.0, max_wait=60.0).build_retrying()
    assert isinstance(retrying.wait, wait_exponential)


def test_build_retrying_fixed():
    retrying = RetryPolicy(backoff="fixed", min_wait=5.0).build_retrying()
    assert isinstance(retrying.wait, wait_fixed)


def test_build_retrying_random_jitter():
    retrying = RetryPolicy(backoff="random_jitter").build_retrying()
    assert isinstance(retrying.wait, wait_random)


def test_build_retrying_stop_after_attempts():
    retrying = RetryPolicy(max_attempts=7).build_retrying()
    assert retrying.stop.max_attempt_number == 7


def test_build_retrying_extra_exceptions():
    class CustomError(Exception):
        pass

    retrying = RetryPolicy().build_retrying(extra_exceptions=[CustomError])
    retry_check = retrying.retry
    assert hasattr(retry_check, "exception_types")
    assert CustomError in retry_check.exception_types
    assert httpx.TimeoutException in retry_check.exception_types
