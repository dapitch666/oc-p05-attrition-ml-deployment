import hashlib
import secrets

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from oc_p05_attrition_ml_deployment.models import ApiKey

KEY_BYTES = 32


def generate_key() -> str:
    """Return a new random key."""
    return secrets.token_urlsafe(KEY_BYTES)


def hash_key(key: str) -> str:
    """Return the SHA-256 hex digest of a key."""
    return hashlib.sha256(key.encode()).hexdigest()


def create_api_key(session: Session, client_name: str) -> tuple[int, str]:
    """Store a new key for a client and return its id and the key, which is not kept."""
    key = generate_key()
    record = ApiKey(client_name=client_name, key_hash=hash_key(key))
    session.add(record)
    session.flush()
    key_id = record.id
    session.commit()
    return key_id, key


def find_active_key_id(session: Session, key: str) -> int | None:
    """Return the id of the non-revoked key matching this value, or None."""
    statement = select(ApiKey.id).where(
        ApiKey.key_hash == hash_key(key), ApiKey.revoked_at.is_(None)
    )
    return session.scalar(statement)


def revoke_api_key(session: Session, key_id: int) -> bool:
    """Revoke a key; return False when it does not exist or is already revoked."""
    result = session.execute(
        update(ApiKey)
        .where(ApiKey.id == key_id, ApiKey.revoked_at.is_(None))
        .values(revoked_at=func.now())
    )
    session.commit()
    return result.rowcount == 1


def has_api_keys(session: Session) -> bool:
    """Return True when at least one key exists, revoked or not."""
    return session.scalar(select(ApiKey.id).limit(1)) is not None


def list_api_keys(session: Session) -> list[ApiKey]:
    """Return every key, ordered by id."""
    return list(session.scalars(select(ApiKey).order_by(ApiKey.id)))
