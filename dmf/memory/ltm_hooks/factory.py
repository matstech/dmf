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

"""Factory for configured long-term-memory hooks."""

from __future__ import annotations

import os

from dmf.models.ltm_hook import LTMHook, NullLTMHook
from dmf.utils.config import VectorConfig
from dmf.utils.config_loader import LTMSettings
from dmf.utils.constants import (
    LTM_BACKEND_CHROMA,
    LTM_BACKEND_FILE,
    LTM_BACKEND_NULL,
    LTM_BACKEND_QDRANT,
)


def build_ltm_hook(settings: LTMSettings, vector_config: VectorConfig) -> LTMHook:
    """Build the configured long-term-memory hook.

    Args:
        settings: LTM backend settings from ``DMFConfig.ltm``.
        vector_config: Vector configuration used by vector-backed hooks.

    Returns:
        Configured LTM hook.

    Raises:
        ValueError: If the backend or a required server credential is invalid.
    """
    if not settings.enabled:
        return NullLTMHook()

    if settings.storage_type == LTM_BACKEND_FILE:
        from dmf.memory.ltm_hooks import FileLTMHook

        return FileLTMHook(
            settings.storage_path,
            cards_enabled=settings.cards_enabled,
            cards_path=settings.cards_path,
        )

    if settings.storage_type == LTM_BACKEND_CHROMA:
        from dmf.memory.ltm_hooks import ChromaLTMHook
        from dmf.memory.ltm_hooks.chroma_client import (
            ChromaConnectionConfig,
            ChromaConnectionMode,
        )

        mode = ChromaConnectionMode(settings.chroma_mode)
        auth_token = _resolve_chroma_auth_token(settings, mode)
        connection = ChromaConnectionConfig(
            mode=mode,
            persist_directory=settings.chroma_path,
            host=settings.chroma_host,
            port=settings.chroma_port,
            ssl=settings.chroma_ssl,
            tenant=settings.chroma_tenant,
            database=settings.chroma_database,
            auth_token=auth_token,
        )
        return ChromaLTMHook(
            collection_name=settings.collection_name,
            persist_directory=settings.chroma_path,
            distance_threshold=settings.distance_threshold,
            vector_config=vector_config,
            cards_enabled=settings.cards_enabled,
            cards_path=settings.cards_path,
            cards_collection_name=settings.cards_collection_name,
            connection=connection,
        )

    if settings.storage_type == LTM_BACKEND_QDRANT:
        from dmf.memory.ltm_hooks import QdrantLTMHook
        from dmf.memory.ltm_hooks.qdrant_client import (
            QdrantConnectionConfig,
            QdrantConnectionMode,
        )

        mode = QdrantConnectionMode(settings.qdrant_mode)
        api_key = _resolve_qdrant_api_key(settings, mode)
        connection = QdrantConnectionConfig(
            mode=mode,
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            ssl=settings.qdrant_ssl,
            api_key=api_key,
            timeout=settings.qdrant_timeout,
        )
        return QdrantLTMHook(
            collection_name=settings.collection_name,
            distance_threshold=settings.distance_threshold,
            vector_config=vector_config,
            cards_enabled=settings.cards_enabled,
            cards_path=settings.cards_path,
            cards_collection_name=settings.cards_collection_name,
            connection=connection,
        )

    if settings.storage_type == LTM_BACKEND_NULL:
        return NullLTMHook()

    raise ValueError(
        "Unsupported ltm.storage_type at runtime: "
        f"{settings.storage_type!r}"
    )


def _resolve_chroma_auth_token(
    settings: LTMSettings,
    mode: object,
) -> str | None:
    from dmf.memory.ltm_hooks.chroma_client import ChromaConnectionMode

    if mode is not ChromaConnectionMode.SERVER:
        return None
    if not settings.chroma_auth_token_env:
        return None

    env_name = settings.chroma_auth_token_env.strip()
    auth_token = os.getenv(env_name)
    if auth_token is None or not auth_token.strip():
        raise ValueError(
            f"Chroma auth token environment variable {env_name!r} "
            "is missing or empty"
        )
    return auth_token


def _resolve_qdrant_api_key(
    settings: LTMSettings,
    mode: object,
) -> str | None:
    from dmf.memory.ltm_hooks.qdrant_client import QdrantConnectionMode

    if mode is not QdrantConnectionMode.SERVER:
        return None
    if not settings.qdrant_api_key_env:
        return None

    env_name = settings.qdrant_api_key_env.strip()
    api_key = os.getenv(env_name)
    if api_key is None or not api_key.strip():
        raise ValueError(
            f"Qdrant API key environment variable {env_name!r} "
            "is missing or empty"
        )
    return api_key
