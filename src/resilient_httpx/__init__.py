from resilient_httpx.client import AsyncProxyHttpClient
from resilient_httpx.exceptions import AllProxiesExhausted, MaxRetriesExceeded
from resilient_httpx.retry import RetryPolicy
from resilient_httpx.sync_client import ProxyHttpClient

__version__ = "0.3.0"

__all__ = [
    "AsyncProxyHttpClient",
    "ProxyHttpClient",
    "RetryPolicy",
    "AllProxiesExhausted",
    "MaxRetriesExceeded",
]
