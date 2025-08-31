"""Authentication schemas."""

from pydantic import BaseModel


class Token(BaseModel):
    """JWT token response schema."""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """JWT token data schema."""
    username: str | None = None
