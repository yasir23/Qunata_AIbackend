"""API module for Open Deep Research."""

from .auth_endpoints import auth_router
from .main import app

__all__ = ["auth_router", "app"]

