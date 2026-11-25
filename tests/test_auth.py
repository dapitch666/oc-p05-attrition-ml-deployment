import hashlib

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from oc_p05_attrition_ml_deployment.auth import (
    create_api_key,
    find_active_key_id,
    generate_key,
    hash_key,
    revoke_api_key,
)
from oc_p05_attrition_ml_deployment.db import get_engine
from oc_p05_attrition_ml_deployment.models import ApiKey


@pytest.fixture
def session():
    with Session(get_engine()) as session:
        yield session


def test_generated_keys_are_long_and_different():
    keys = {generate_key() for _ in range(5)}

    assert len(keys) == 5
    assert all(len(key) >= 40 for key in keys)


def test_hash_key_is_the_sha256_hex_digest():
    assert hash_key("abc") == hashlib.sha256(b"abc").hexdigest()


def test_create_api_key_stores_the_hash_and_returns_the_key(session):
    key_id, key = create_api_key(session, "Client A")

    record = session.get(ApiKey, key_id)
    assert record.client_name == "Client A"
    assert record.key_hash == hash_key(key)
    assert record.key_hash != key
    assert record.revoked_at is None
    assert record.created_at is not None


def test_key_hash_is_unique(session):
    create_api_key(session, "Client A")
    duplicate = ApiKey(
        client_name="Client B", key_hash=session.scalars(select(ApiKey.key_hash)).one()
    )
    session.add(duplicate)

    with pytest.raises(IntegrityError):
        session.commit()


def test_find_active_key_id_matches_only_the_right_key(session):
    key_id, key = create_api_key(session, "Client A")

    assert find_active_key_id(session, key) == key_id
    assert find_active_key_id(session, key + "x") is None
    assert find_active_key_id(session, "") is None


def test_revoked_key_is_no_longer_found(session):
    key_id, key = create_api_key(session, "Client A")

    assert revoke_api_key(session, key_id) is True

    assert find_active_key_id(session, key) is None
    session.expire_all()
    assert session.get(ApiKey, key_id).revoked_at is not None


def test_revoking_twice_or_an_unknown_id_returns_false(session):
    key_id, _ = create_api_key(session, "Client A")
    revoke_api_key(session, key_id)

    assert revoke_api_key(session, key_id) is False
    assert revoke_api_key(session, key_id + 100) is False
