# Copyright (c) 2026-present matstech
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT

"""Construction of embedded and server-backed Chroma clients."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.shared_system_client import SharedSystemClient
from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT, Settings

from dmf.memory.ltm_hooks.chroma_retry import RetryingFastAPI
from dmf.utils.constants import DEFAULT_LTM_CHROMA_PATH

_BACKEND_INSTALL_LOCK = threading.RLock()
_RETRYING_FASTAPI_FQN = f"{RetryingFastAPI.__module__}.{RetryingFastAPI.__qualname__}"


class ChromaConnectionMode(str, Enum):
    """Supported Chroma deployment modes."""

    EMBEDDED = "embedded"
    SERVER = "server"


@dataclass(frozen=True)
class ChromaConnectionConfig:
    """Connection parameters used by :func:`build_chroma_client`."""

    mode: ChromaConnectionMode = ChromaConnectionMode.EMBEDDED
    persist_directory: Path | str = DEFAULT_LTM_CHROMA_PATH
    host: str = "localhost"
    port: int = 8000
    ssl: bool = False
    tenant: str = DEFAULT_TENANT
    database: str = DEFAULT_DATABASE
    auth_token: str | None = field(default=None, repr=False)


def build_chroma_client(connection: ChromaConnectionConfig) -> ClientAPI:
    """Build a Chroma client for the requested deployment mode."""
    if connection.mode == ChromaConnectionMode.EMBEDDED:
        persist_path = Path(connection.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(anonymized_telemetry=False),
            tenant=connection.tenant,
            database=connection.database,
        )

    if connection.mode == ChromaConnectionMode.SERVER:
        headers = None
        if connection.auth_token:
            headers = {"Authorization": f"Bearer {connection.auth_token}"}

        settings = Settings(
            anonymized_telemetry=False,
            chroma_api_impl=_RETRYING_FASTAPI_FQN,
            chroma_server_host=connection.host,
            chroma_server_http_port=connection.port,
            chroma_server_ssl_enabled=connection.ssl,
            chroma_server_headers=headers,
        )
        with _allow_retrying_backend():
            return chromadb.Client(
                settings=settings,
                tenant=connection.tenant,
                database=connection.database,
            )

    raise ValueError(f"Unsupported Chroma connection mode: {connection.mode!r}")


@contextmanager
def _allow_retrying_backend() -> Iterator[None]:
    """Temporarily allow the DMF backend in Chroma 0.6.3's identifier resolver."""
    with _BACKEND_INSTALL_LOCK:
        original_descriptor = SharedSystemClient.__dict__["_get_identifier_from_settings"]
        original_resolver = SharedSystemClient._get_identifier_from_settings

        def resolve_identifier(settings: Settings) -> str:
            if settings.chroma_api_impl == _RETRYING_FASTAPI_FQN:
                return str(uuid.uuid4())
            return original_resolver(settings)

        SharedSystemClient._get_identifier_from_settings = staticmethod(resolve_identifier)
        try:
            yield
        finally:
            SharedSystemClient._get_identifier_from_settings = original_descriptor


__all__ = [
    "ChromaConnectionConfig",
    "ChromaConnectionMode",
    "build_chroma_client",
]
