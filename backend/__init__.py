"""FastAPI backend package for the m_photo geo-location app.

The ASGI application object lives at ``backend.app.main:app`` so the container
entrypoint can stay::

    uv run --frozen uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
