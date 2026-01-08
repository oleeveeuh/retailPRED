"""
API Module for RetailPRED Backend
Contains FastAPI routes and schemas
"""

import sys
from pathlib import Path

# Add app directory to Python path
app_path = Path(__file__).parent.parent
if str(app_path) not in sys.path:
    sys.path.insert(0, str(app_path))

from .routes import router
from .schemas import *

__all__ = ["router"]
