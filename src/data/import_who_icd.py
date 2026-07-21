from __future__ import annotations
import os

def credentials_available() -> bool:
    return bool(os.environ.get('WHO_ICD_CLIENT_ID') and os.environ.get('WHO_ICD_CLIENT_SECRET'))

def require_credentials():
    if not credentials_available():
        raise RuntimeError('WHO ICD credentials are required via environment variables')
