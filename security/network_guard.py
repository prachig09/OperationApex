"""Local Processing Guarantee.

Process-wide egress guard. While active it intercepts socket ``connect``,
``connect_ex``, ``sendto`` and DNS resolution (``getaddrinfo`` /
``gethostbyname``). Any attempt to reach a non-permitted host raises
``NetworkViolation`` and is recorded in a ``NetworkAudit``. With
``strict_exit=True`` the context manager raises on exit if *any* attempt was
blocked, even if the offending library swallowed the exception.
"""
from __future__ import annotations

import contextlib
import functools
import ipaddress
import os
import socket
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

OFFLINE_ENV: Dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "ANONYMIZED_TELEMETRY": "False",
    "CHROMA_TELEMETRY": "False",
    "DO_NOT_TRACK": "true",
    "SCARF_NO_ANALYTICS": "true",
    "NO_PROXY": "*",
}


def enforce_offline_environment() -> None:
    """Force offline/no-telemetry flags for HuggingFace, Chroma and Unstructured."""
    for key, value in OFFLINE_ENV.items():
        os.environ[key] = value
    for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(proxy_var, None)


enforce_offline_environment()  # must run before transformers / huggingface_hub import


class NetworkViolation(RuntimeError):
    """Raised when guarded code attempts a non-permitted network operation."""


@dataclass(eq=False)
class NetworkAudit:
    """Record of network operations observed inside a guarded block."""

    allow_loopback: bool
    allowed_hosts: frozenset
    permitted: List[str] = field(default_factory=list)
    blocked: List[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """True when no operation was blocked."""
        return not self.blocked

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"permitted(loopback/allow-listed)={len(self.permitted)} "
            f"blocked={len(self.blocked)} clean={self.clean}"
        )


_LOCK = threading.RLock()
_STACK: List[NetworkAudit] = []
_ORIGINALS: Dict[str, Any] = {}


def _host_text(host: Any) -> str:
    if isinstance(host, (bytes, bytearray)):
        host = host.decode("utf-8", "ignore")
    host = str(host).strip().strip("[]")
    return host.split("%", 1)[0].lower()


def _is_permitted(host: Any, audit: NetworkAudit) -> bool:
    if host is None or host == "":
        return True  # wildcard / passive operations never leave the machine
    text = _host_text(host)
    if text in audit.allowed_hosts:
        return True
    if not audit.allow_loopback:
        return False
    if text in {"localhost", "172.17.0.1", "host.docker.internal"}:
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


def _check(kind: str, host: Any) -> None:
    with _LOCK:
        audit = _STACK[-1] if _STACK else None
    if audit is None:
        return
    label = f"{kind}:{host!r}"
    if _is_permitted(host, audit):
        audit.permitted.append(label)
        return
    audit.blocked.append(label)
    raise NetworkViolation(f"External network access blocked by Local Processing Guarantee ({label})")


def _address_host(address: Any) -> Any:
    # AF_UNIX addresses are plain strings/bytes: local IPC, never external.
    if isinstance(address, tuple) and address:
        return address[0]
    return None


def _install() -> None:
    _ORIGINALS.update(
        connect=socket.socket.connect,
        connect_ex=socket.socket.connect_ex,
        sendto=socket.socket.sendto,
        getaddrinfo=socket.getaddrinfo,
        gethostbyname=socket.gethostbyname,
    )

    def connect(self, address):
        _check("connect", _address_host(address))
        return _ORIGINALS["connect"](self, address)

    def connect_ex(self, address):
        _check("connect_ex", _address_host(address))
        return _ORIGINALS["connect_ex"](self, address)

    def sendto(self, data, *args):
        _check("sendto", _address_host(args[-1]) if args else None)
        return _ORIGINALS["sendto"](self, data, *args)

    def getaddrinfo(host, *args, **kwargs):
        _check("dns", host)
        return _ORIGINALS["getaddrinfo"](host, *args, **kwargs)

    def gethostbyname(host):
        _check("dns", host)
        return _ORIGINALS["gethostbyname"](host)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.socket.sendto = sendto
    socket.getaddrinfo = getaddrinfo
    socket.gethostbyname = gethostbyname


def _uninstall() -> None:
    socket.socket.connect = _ORIGINALS["connect"]
    socket.socket.connect_ex = _ORIGINALS["connect_ex"]
    socket.socket.sendto = _ORIGINALS["sendto"]
    socket.getaddrinfo = _ORIGINALS["getaddrinfo"]
    socket.gethostbyname = _ORIGINALS["gethostbyname"]
    _ORIGINALS.clear()


@contextlib.contextmanager
def local_processing_guarantee(
    allowed_hosts: Iterable[str] = (),
    allow_loopback: bool = True,
    strict_exit: bool = True,
) -> Iterator[NetworkAudit]:
    """Guarantee that the enclosed block performs no external network calls.

    Args:
        allowed_hosts: Extra explicitly permitted hosts (e.g. a LAN database).
        allow_loopback: Permit localhost/127.0.0.0/8/::1. Use False for
            parsing/chunking/embedding, which must open zero sockets.
        strict_exit: Raise ``NetworkViolation`` on exit if anything was blocked.

    Yields:
        The live ``NetworkAudit`` for the block.

    Raises:
        NetworkViolation: On any blocked attempt (immediately and/or on exit).
    """
    enforce_offline_environment()
    audit = NetworkAudit(allow_loopback, frozenset(h.lower() for h in allowed_hosts))
    with _LOCK:
        if not _STACK:
            _install()
        _STACK.append(audit)
    try:
        yield audit
    finally:
        with _LOCK:
            _STACK.remove(audit)
            if not _STACK:
                _uninstall()
    if strict_exit and audit.blocked:
        raise NetworkViolation(
            "Local Processing Guarantee violated; blocked attempts: " + ", ".join(audit.blocked)
        )


def guaranteed_local_call(
    func: Callable[..., Any],
    *args: Any,
    allowed_hosts: Iterable[str] = (),
    allow_loopback: bool = False,
    **kwargs: Any,
) -> Tuple[Any, NetworkAudit]:
    """Run ``func`` under the guarantee and return ``(result, audit)``."""
    with local_processing_guarantee(allowed_hosts, allow_loopback) as audit:
        result = func(*args, **kwargs)
    return result, audit


def enforce_local(allowed_hosts: Iterable[str] = (), allow_loopback: bool = False) -> Callable:
    """Decorator form of :func:`local_processing_guarantee`."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with local_processing_guarantee(allowed_hosts, allow_loopback):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def verify_guard_blocks_egress() -> bool:
    """Self-test: confirm the guard actually blocks external IP and DNS egress.

    Returns:
        True if both probes were blocked before any packet could be sent.
    """
    probes = [("203.0.113.10", 443), ("example.invalid", 80)]
    with local_processing_guarantee(allow_loopback=True, strict_exit=False) as audit:
        for host, port in probes:
            try:
                socket.create_connection((host, port), timeout=0.5)
            except NetworkViolation:
                continue
            except OSError:
                return False
            return False
    return len(audit.blocked) >= len(probes)