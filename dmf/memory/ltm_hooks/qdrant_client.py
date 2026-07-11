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

"""Construction of Qdrant clients for the LTM backend."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

_QDRANT_EXTRA_INSTALL_MESSAGE = (
    "Install the Qdrant backend with: pip install 'dmf-memory[qdrant]'"
)


class QdrantConnectionMode(str, Enum):
    """Supported Qdrant deployment modes."""

    MEMORY = "memory"


@dataclass(frozen=True)
class QdrantConnectionConfig:
    """Connection parameters used by :func:`build_qdrant_client`."""

    mode: QdrantConnectionMode = QdrantConnectionMode.MEMORY


def build_qdrant_client(connection: QdrantConnectionConfig) -> object:
    """Build a Qdrant client for the requested deployment mode."""
    if connection.mode == QdrantConnectionMode.MEMORY:
        try:
            from qdrant_client import QdrantClient  # noqa: PLC0415
        except ModuleNotFoundError as exc:
            if exc.name == "qdrant_client":
                raise ModuleNotFoundError(_QDRANT_EXTRA_INSTALL_MESSAGE) from exc
            raise

        return QdrantClient(":memory:")

    raise ValueError(f"Unsupported Qdrant connection mode: {connection.mode!r}")


__all__ = [
    "QdrantConnectionConfig",
    "QdrantConnectionMode",
    "build_qdrant_client",
]
