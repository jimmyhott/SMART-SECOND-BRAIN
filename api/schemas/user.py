"""User schemas."""

from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    username: str


class UserCreate(UserBase):
    """User creation schema."""
    password: str


class UserUpdate(BaseModel):
    """User update schema."""
    email: EmailStr | None = None
    username: str | None = None
    password: str | None = None


class UserResponse(UserBase):
    """User response schema."""
    id: int
    is_active: bool = True

    class Config:
        """Pydantic configuration."""
        from_attributes = True
