from resilient_httpx.client import ProxyHttpClient
from resilient_httpx.exceptions import AllProxiesExhausted, MaxRetriesExceeded
from resilient_httpx.retry import RetryPolicy

__version__ = "0.1.0"

__all__ = [
    "ProxyHttpClient",
    "RetryPolicy",
    "AllProxiesExhausted",
    "MaxRetriesExceeded",
]
