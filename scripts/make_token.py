"""Create a bearer token for a user: prints the token once (store it in a secret manager or
hand it to the user) and the hash to put in the users file. The token itself is never saved.

Usage:
    uv run python scripts/make_token.py
"""

import secrets

from wfx.api.auth import hash_token


def main() -> None:
    token = secrets.token_urlsafe(32)
    print(f"token (give to the user, then discard): {token}")
    print(f"token_sha256 (put in the users file):   {hash_token(token)}")


if __name__ == "__main__":
    main()
