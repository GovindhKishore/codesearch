import keyring

SERVICE_NAME = "codesearch"

def get_api_key(provider: str) -> str | None:
    try:
        return keyring.get_password(SERVICE_NAME, provider)
    except Exception:
        return None


def set_api_key(provider: str, api_key: str) -> bool:
    try:
        keyring.set_password(SERVICE_NAME, provider, api_key)
        return True
    except Exception:
        return False

def clear_api_key(provider: str) -> bool:
    try:
        keyring.delete_password(SERVICE_NAME, provider)
        return True
    except Exception:
        return False