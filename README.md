# resilient-httpx

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)

HTTP client with retry, backoff, and proxy rotation. Sync and async. Built on top of `httpx`, `tenacity`, and `structlog`.

## Installation

```bash
pip install resilient-httpx
```

## Usage

### Sync client

```python
from resilient_httpx import ProxyHttpClient, RetryPolicy

with ProxyHttpClient(retry=RetryPolicy(max_attempts=5), timeout=15.0) as client:
    response = client.get("https://api.example.com/data")
```

### Async client

```python
from resilient_httpx import AsyncProxyHttpClient, RetryPolicy

async with AsyncProxyHttpClient(retry=RetryPolicy(max_attempts=5), timeout=15.0) as client:
    response = await client.get("https://api.example.com/data")
```

### With proxy rotation

```python
from resilient_httpx import ProxyHttpClient, RetryPolicy

client = ProxyHttpClient(
    proxies=[
        "http://proxy1:8080",
        "http://user:pass@proxy2:8080",
        "socks5://proxy3:1080",
    ],
    proxy_strategy="round-robin",  # or "random"
    retry=RetryPolicy(max_attempts=5, backoff="exponential"),
    timeout=15.0,
    blacklist_threshold=3,
    blacklist_ttl=300.0,
)

with client:
    response = client.get("https://api.example.com/data")
    data = response.json()
```

`AsyncProxyHttpClient` accepts the same parameters:

```python
from resilient_httpx import AsyncProxyHttpClient, RetryPolicy

async with AsyncProxyHttpClient(
    proxies=["http://proxy1:8080", "http://proxy2:8080"],
    retry=RetryPolicy(max_attempts=5, backoff="exponential"),
) as client:
    response = await client.get("https://api.example.com/data")
```

### Custom retry policy

```python
import httpx
from resilient_httpx import RetryPolicy

policy = RetryPolicy(
    max_attempts=5,
    backoff="exponential",       # "exponential" | "fixed" | "random_jitter"
    min_wait=1.0,
    max_wait=30.0,
    retry_on=[429, 500, 502, 503, 504],
    retry_on_exception=[httpx.TimeoutException, httpx.ConnectError],
)
```

### Error handling

```python
from resilient_httpx import ProxyHttpClient, AllProxiesExhausted, MaxRetriesExceeded

with ProxyHttpClient(proxies=proxy_list) as client:
    try:
        response = client.get("https://api.example.com/data")
    except AllProxiesExhausted:
        ...  # all proxies blacklisted
    except MaxRetriesExceeded as exc:
        ...  # retries exhausted, exc.__cause__ has the last error
```

## Configuration Reference

Both `ProxyHttpClient` and `AsyncProxyHttpClient` accept the same parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `proxies` | `list[str] \| None` | `None` | Proxy URLs |
| `proxy_strategy` | `str` | `"round-robin"` | `"round-robin"` or `"random"` |
| `retry` | `RetryPolicy` | `RetryPolicy()` | Retry configuration |
| `timeout` | `float` | `30.0` | Request timeout in seconds |
| `headers` | `dict[str, str] \| None` | `None` | Default headers |
| `blacklist_threshold` | `int` | `3` | Consecutive failures before blacklisting |
| `blacklist_ttl` | `float` | `300.0` | Blacklist duration in seconds |
| `fallback_to_direct` | `bool` | `False` | Request without proxy when all are blacklisted |

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Run with coverage:

```bash
pytest --cov=resilient_httpx --cov-report=term-missing
```
