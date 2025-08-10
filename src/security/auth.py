import os
import asyncio
from langgraph_sdk import Auth
from langgraph_sdk.auth.types import StudioUser
from typing import Optional, Any
from .firebase_auth import firebase_service
import logging

logger = logging.getLogger(__name__)

# The "Auth" object is a container that LangGraph will use to mark our authentication function
auth = Auth()


# The `authenticate` decorator tells LangGraph to call this function as middleware
# for every request. This will determine whether the request is allowed or not
@auth.authenticate
async def get_current_user(authorization: str | None) -> Auth.types.MinimalUserDict:
    """Check if the user's JWT token is valid using Firebase Authentication."""

    # Ensure we have authorization header
    if not authorization:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Authorization header missing"
        )

    # Parse the authorization header
    try:
        scheme, token = authorization.split()
        assert scheme.lower() == "bearer"
    except (ValueError, AssertionError):
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Invalid authorization header format"
        )

    # Ensure Firebase service is initialized
    if not firebase_service.is_initialized():
        raise Auth.exceptions.HTTPException(
            status_code=500, detail="Firebase authentication service not initialized"
        )

    try:
        # Verify the Firebase ID token
        user_info = await firebase_service.verify_token(token)

        if not user_info:
            raise Auth.exceptions.HTTPException(
                status_code=401, detail="Invalid token or user not found"
            )

        # Store/update user profile in Firestore (async, don't block on this)
        profile_data = {
            'email': user_info.get('email'),
            'name': user_info.get('name'),
            'picture': user_info.get('picture'),
            'email_verified': user_info.get('email_verified', False),
            'provider_data': user_info.get('provider_data', []),
            'last_login': None  # Will be set to SERVER_TIMESTAMP in Firestore
        }
        
        # Store profile asynchronously (don't block on this)
        asyncio.create_task(
            firebase_service.store_user_profile(user_info['uid'], profile_data)
        )

        # Return user info compatible with LangGraph Auth
        return {
            "identity": user_info['uid'],
        }
        
    except Auth.exceptions.HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Handle any other errors from Firebase
        logger.error(f"Firebase authentication error: {e}")
        raise Auth.exceptions.HTTPException(
            status_code=401, detail=f"Authentication failed: {str(e)}"
        )


@auth.on.threads.create
@auth.on.threads.create_run
async def on_thread_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.create.value,
):
    """Add owner when creating threads.

    This handler runs when creating new threads and does two things:
    1. Sets metadata on the thread being created to track ownership
    2. Returns a filter that ensures only the creator can access it
    """

    if isinstance(ctx.user, StudioUser):
        return

    # Add owner metadata to the thread being created
    # This metadata is stored with the thread and persists
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity


@auth.on.threads.read
@auth.on.threads.delete
@auth.on.threads.update
@auth.on.threads.search
async def on_thread_read(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.read.value,
):
    """Only let users read their own threads.

    This handler runs on read operations. We don't need to set
    metadata since the thread already exists - we just need to
    return a filter to ensure users can only see their own threads.
    """
    if isinstance(ctx.user, StudioUser):
        return

    return {"owner": ctx.user.identity}


@auth.on.assistants.create
async def on_assistants_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.assistants.create.value,
):
    if isinstance(ctx.user, StudioUser):
        return

    # Add owner metadata to the assistant being created
    # This metadata is stored with the assistant and persists
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity


@auth.on.assistants.read
@auth.on.assistants.delete
@auth.on.assistants.update
@auth.on.assistants.search
async def on_assistants_read(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.assistants.read.value,
):
    """Only let users read their own assistants.

    This handler runs on read operations. We don't need to set
    metadata since the assistant already exists - we just need to
    return a filter to ensure users can only see their own assistants.
    """

    if isinstance(ctx.user, StudioUser):
        return

    return {"owner": ctx.user.identity}


@auth.on.store()
async def authorize_store(ctx: Auth.types.AuthContext, value: dict):
    if isinstance(ctx.user, StudioUser):
        return

    # The "namespace" field for each store item is a tuple you can think of as the directory of an item.
    namespace: tuple = value["namespace"]
    assert namespace[0] == ctx.user.identity, "Not authorized"

