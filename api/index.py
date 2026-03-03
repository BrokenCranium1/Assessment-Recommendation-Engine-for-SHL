import sys
import os

# Add the project root to the path so we can import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the FastAPI app from your main application
from main import app

# Vercel expects an ASGI application to be named 'app'
# This 'app' is the same one from your main.py