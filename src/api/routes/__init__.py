"""
API routes package for ON-CHAIN SUPER SIGNALS™.

Exports all route modules for registration in the main FastAPI application.
"""

from src.api.routes import alerts, chains, signals

__all__ = ["alerts", "chains", "signals"]
