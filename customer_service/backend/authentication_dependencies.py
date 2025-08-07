# auth_dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Annotated, Optional
import httpx
from pydantic import BaseModel, Field

from config import AUTH_PROFILE_ENDPOINT, AUTH_LOGIN_URL

# Define a Pydantic model for the user profile returned by your auth service
class UserProfile(BaseModel):
    id: str
    email: str
    # Add other profile fields as needed

# This is what FastAPI uses to expect a Bearer token in the Authorization header.
# The tokenUrl is primarily for OpenAPI/Swagger UI to know where to send credentials,
# but our actual validation logic will be custom.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=AUTH_LOGIN_URL)

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> UserProfile:
    """
    Dependency to get the current authenticated user.
    If the token is invalid, it raises an HTTPException.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                AUTH_PROFILE_ENDPOINT,
                headers={"Authorization": f"Bearer {token}"}
            )
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            user_profile = UserProfile(**response.json())
            return user_profile
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                # If the external API explicitly says Unauthorized, return 401 to client
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated",
                    headers={"WWW-Authenticate": f"Bearer realm='{AUTH_LOGIN_URL}'"},
                )
            elif e.response.status_code == 403:
                # If the external API explicitly says Forbidden, return 403 to client
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not authorized to access this resource",
                )
            else:
                # Handle other HTTP errors from the auth service
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Authentication service error: {e.response.status_code} {e.response.text}",
                )
        except httpx.RequestError as e:
            # Handle network or connection errors to the auth service
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Could not connect to authentication service: {e}",
            )
        except Exception as e:
            # Catch any other unexpected errors during token validation
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred during authentication: {e}",
            )

# Dependency for optional authentication
from fastapi import Request

async def get_optional_current_user(request: Request) -> Optional[UserProfile]:
    """
    Dependency to get the current authenticated user, but allows unauthenticated access.
    Returns None if no valid token is provided.
    """
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        return None

    token = auth_header.split(" ", 1)[1]
    try:
        user = await get_current_user(token)
        return user
    except HTTPException as e:
        if e.status_code == status.HTTP_401_UNAUTHORIZED:
            return None
        raise