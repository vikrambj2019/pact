"""Storage abstraction — local filesystem for Phase 1, swap to S3 later."""
import hashlib
import uuid
from pathlib import Path
from .config import settings


class LocalStorage:
    def __init__(self, base: Path):
        self.base = base
        self.base.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Two-level prefix to avoid flat directory explosion
        return self.base / key[:2] / key[2:4] / key

    async def put(self, data: bytes, suffix: str = "") -> tuple[str, str]:
        """Store bytes, return (storage_key, sha256_hex)."""
        digest = hashlib.sha256(data).hexdigest()
        key = f"{uuid.uuid4().hex}{suffix}"
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return key, digest

    async def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    async def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()


def get_storage() -> LocalStorage:
    return LocalStorage(Path(settings.storage_local_path))
