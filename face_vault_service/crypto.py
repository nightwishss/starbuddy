"""ChaCha20-Poly1305 AEAD with an optional optimized backend."""

from __future__ import annotations

import hmac
import os
import struct
from pathlib import Path


KEY_BYTES = 32
NONCE_BYTES = 12
TAG_BYTES = 16


class CryptoError(RuntimeError):
    pass


def load_or_create_key(path: str | Path) -> bytes:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        key = path.read_bytes()
    else:
        try:
            key = os.urandom(KEY_BYTES)
            os.write(fd, key)
            os.fsync(fd)
        finally:
            os.close(fd)
    os.chmod(path, 0o600)
    if len(key) != KEY_BYTES:
        raise CryptoError("face-vault key must contain exactly 32 bytes")
    return key


def _rotate(value: int, count: int) -> int:
    return ((value << count) & 0xFFFFFFFF) | (value >> (32 - count))


def _quarter(state: list[int], a: int, b: int, c: int, d: int) -> None:
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF; state[d] ^= state[a]; state[d] = _rotate(state[d], 16)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF; state[b] ^= state[c]; state[b] = _rotate(state[b], 12)
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF; state[d] ^= state[a]; state[d] = _rotate(state[d], 8)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF; state[b] ^= state[c]; state[b] = _rotate(state[b], 7)


def _block(key: bytes, counter: int, nonce: bytes) -> bytes:
    initial = list(struct.unpack("<4I", b"expand 32-byte k") + struct.unpack("<8I", key) +
                   (counter,) + struct.unpack("<3I", nonce))
    state = initial.copy()
    for _ in range(10):
        _quarter(state, 0, 4, 8, 12); _quarter(state, 1, 5, 9, 13)
        _quarter(state, 2, 6, 10, 14); _quarter(state, 3, 7, 11, 15)
        _quarter(state, 0, 5, 10, 15); _quarter(state, 1, 6, 11, 12)
        _quarter(state, 2, 7, 8, 13); _quarter(state, 3, 4, 9, 14)
    return struct.pack("<16I", *((state[index] + initial[index]) & 0xFFFFFFFF for index in range(16)))


def _stream_xor(key: bytes, nonce: bytes, value: bytes) -> bytes:
    output = bytearray(len(value))
    for offset in range(0, len(value), 64):
        stream = _block(key, 1 + offset // 64, nonce)
        chunk = value[offset:offset + 64]
        output[offset:offset + len(chunk)] = bytes(a ^ b for a, b in zip(chunk, stream))
    return bytes(output)


def _pad16(value: bytes) -> bytes:
    return value + b"\0" * ((-len(value)) % 16)


def _poly1305(key: bytes, message: bytes) -> bytes:
    r = int.from_bytes(key[:16], "little") & 0x0FFFFFFC0FFFFFFC0FFFFFFC0FFFFFFF
    s = int.from_bytes(key[16:], "little")
    accumulator, prime = 0, (1 << 130) - 5
    for offset in range(0, len(message), 16):
        block = message[offset:offset + 16]
        accumulator = ((accumulator + int.from_bytes(block + b"\x01", "little")) * r) % prime
    return ((accumulator + s) % (1 << 128)).to_bytes(16, "little")


def _tag(key: bytes, nonce: bytes, ciphertext: bytes, associated_data: bytes) -> bytes:
    one_time_key = _block(key, 0, nonce)[:32]
    mac_data = (_pad16(associated_data) + _pad16(ciphertext) +
                struct.pack("<QQ", len(associated_data), len(ciphertext)))
    return _poly1305(one_time_key, mac_data)


class AuthenticatedCipher:
    """RFC 8439 ChaCha20-Poly1305 with ciphertext followed by its 16-byte tag."""

    def __init__(self, key: bytes):
        if len(key) != KEY_BYTES:
            raise ValueError("ChaCha20-Poly1305 requires a 32-byte key")
        self.key = key
        try:
            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        except ImportError:
            self._backend = None
        else:
            self._backend = ChaCha20Poly1305(key)

    def encrypt(self, nonce: bytes, plaintext: bytes, associated_data: bytes) -> bytes:
        if len(nonce) != NONCE_BYTES:
            raise ValueError("invalid AEAD nonce")
        if self._backend:
            return self._backend.encrypt(nonce, plaintext, associated_data)
        ciphertext = _stream_xor(self.key, nonce, plaintext)
        return ciphertext + _tag(self.key, nonce, ciphertext, associated_data)

    def decrypt(self, nonce: bytes, ciphertext: bytes, associated_data: bytes) -> bytes:
        if len(nonce) != NONCE_BYTES or len(ciphertext) < TAG_BYTES:
            raise CryptoError("invalid encrypted template")
        if self._backend:
            try:
                return self._backend.decrypt(nonce, ciphertext, associated_data)
            except Exception as exc:
                raise CryptoError("ciphertext authentication failed") from exc
        body, actual_tag = ciphertext[:-TAG_BYTES], ciphertext[-TAG_BYTES:]
        expected_tag = _tag(self.key, nonce, body, associated_data)
        if not hmac.compare_digest(actual_tag, expected_tag):
            raise CryptoError("ciphertext authentication failed")
        return _stream_xor(self.key, nonce, body)
