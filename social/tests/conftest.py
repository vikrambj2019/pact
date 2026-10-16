import sys
from pathlib import Path

# Add social/ to path so tests can import pipeline modules directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
