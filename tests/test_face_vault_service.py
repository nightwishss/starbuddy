import math
import os
import sqlite3
import struct

import pytest

from face_vault_service.crypto import AuthenticatedCipher, CryptoError
from face_vault_service.service import dispatch
from face_vault_service.store import FaceVaultError, FaceVaultStore


@pytest.fixture
def vault(tmp_path):
    return FaceVaultStore(tmp_path / "data" / "vault.sqlite3", tmp_path / "secrets" / "vault.key", clock=lambda: 1234.5)


def create(vault):
    return vault.identity_create(consent="user_confirmed", consent_version="m3-test", profile_id="profile-test", label="Test")


def test_rfc_8439_aead_vector_and_tamper_rejection():
    key = bytes.fromhex("808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f")
    nonce = bytes.fromhex("070000004041424344454647")
    aad = bytes.fromhex("50515253c0c1c2c3c4c5c6c7")
    plaintext = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it."
    expected = bytes.fromhex(
        "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d6"
        "3dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b36"
        "92ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc"
        "3ff4def08e4b7a9de576d26586cec64b61161ae10b594f09e26a7e902ecbd0600691"
    )
    cipher = AuthenticatedCipher(key)
    encrypted = cipher.encrypt(nonce, plaintext, aad)
    assert encrypted == expected
    assert cipher.decrypt(nonce, encrypted, aad) == plaintext
    with pytest.raises(CryptoError):
        cipher.decrypt(nonce, encrypted[:-1] + bytes([encrypted[-1] ^ 1]), aad)


def test_empty_vault_and_private_permissions(vault):
    assert vault.status()["templates"] == 0
    assert vault.status()["matching_enabled"] is True
    assert os.stat(vault.path).st_mode & 0o777 == 0o600
    assert os.stat(vault.key_path).st_mode & 0o777 == 0o600
    assert os.stat(vault.path.parent).st_mode & 0o777 == 0o700
    assert os.stat(vault.key_path.parent).st_mode & 0o777 == 0o700


def test_create_requires_explicit_consent(vault):
    with pytest.raises(FaceVaultError, match="confirmation") as error:
        vault.identity_create(consent="implicit", consent_version="v1")
    assert error.value.code == "consent_required"
    identity = create(vault)
    assert identity["profile_id"] == "profile-test"
    assert identity["templates"] == []


def test_template_is_encrypted_and_metadata_only(vault):
    identity = create(vault)
    vector = [index / 100 for index in range(32)]
    template = vault.template_add(identity["identity_id"], "arcface-test-v1", vector, consent="user_confirmed", quality=.9)
    returned = vault.identity_get(identity["identity_id"])["templates"][0]
    assert returned == template
    assert "embedding" not in returned and "ciphertext" not in returned and "nonce" not in returned
    db = sqlite3.connect(vault.path)
    nonce, ciphertext = db.execute("SELECT nonce,ciphertext FROM templates").fetchone()
    assert len(nonce) == 12 and len(ciphertext) == 32 * 4 + 16
    assert struct.pack("<32f", *vector) not in bytes(ciphertext)
    decoded = vault._decrypt_template_for_matching(template["template_id"])
    assert decoded == pytest.approx(vector, abs=1e-6)


def test_invalid_embeddings_are_rejected(vault):
    identity = create(vault)
    for vector in ([0.0] * 31, [0.0] * 2049, [0.0] * 31 + [math.nan]):
        with pytest.raises(FaceVaultError) as error:
            vault.template_add(identity["identity_id"], "model-v1", vector, consent="user_confirmed")
        assert error.value.code == "invalid_embedding"


def test_revoke_erases_templates_and_blocks_new_ones(vault):
    identity = create(vault); identity_id = identity["identity_id"]
    vault.template_add(identity_id, "model-v1", [0.1] * 32, consent="user_confirmed")
    result = vault.identity_revoke(identity_id, consent="user_confirmed")
    assert result["removed_templates"] == 1
    assert vault.identity_get(identity_id)["status"] == "revoked"
    with pytest.raises(FaceVaultError) as error:
        vault.template_add(identity_id, "model-v1", [0.1] * 32, consent="user_confirmed")
    assert error.value.code == "identity_revoked"


def test_delete_reset_audit_and_raw_media_rejection(vault):
    first, second = create(vault), create(vault)
    vault.identity_delete(first["identity_id"], consent="guardian_confirmed")
    with pytest.raises(FaceVaultError) as error:
        vault.vault_reset(consent="user_confirmed", confirmation="yes")
    assert error.value.code == "confirmation_required"
    reset = vault.vault_reset(consent="user_confirmed", confirmation="DELETE_ALL_BIOMETRICS")
    assert reset["removed_identities"] == 1 and second["identity_id"]
    assert len(vault.audit_list()["items"]) >= 4
    with pytest.raises(FaceVaultError) as error:
        dispatch(vault, {"operation": "template.add", "image": "raw"})
    assert error.value.code == "raw_media_rejected"


def test_individual_template_delete_requires_consent(vault):
    identity = create(vault)
    template = vault.template_add(identity["identity_id"], "model-v1", [0.2] * 32, consent="user_confirmed")
    with pytest.raises(FaceVaultError) as error:
        vault.template_delete(template["template_id"], consent="implicit")
    assert error.value.code == "consent_required"
    result = vault.template_delete(template["template_id"], consent="user_confirmed")
    assert result["deleted"] is True
    assert vault.identity_get(identity["identity_id"])["templates"] == []


def test_match_is_thresholded_and_unknown_by_default(vault):
    assert vault.match("sface-v1", [0.1] * 32)["reason"] == "no_compatible_templates"
    identity = create(vault)
    vault.template_add(identity["identity_id"], "sface-v1", [1.0] + [0.0] * 31,
                       consent="user_confirmed")
    known = vault.match("sface-v1", [0.99, 0.01] + [0.0] * 30)
    assert known["status"] == "known" and known["identity_id"] == identity["identity_id"]
    unknown = vault.match("sface-v1", [0.0, 1.0] + [0.0] * 30)
    assert unknown["status"] == "unknown" and unknown["reason"] == "below_threshold"


def test_match_rejects_ambiguous_identity(vault):
    first, second = create(vault), create(vault)
    vector = [1.0] + [0.0] * 31
    vault.template_add(first["identity_id"], "sface-v1", vector, consent="user_confirmed")
    vault.template_add(second["identity_id"], "sface-v1", vector, consent="user_confirmed")
    assert vault.match("sface-v1", vector)["reason"] == "ambiguous"
