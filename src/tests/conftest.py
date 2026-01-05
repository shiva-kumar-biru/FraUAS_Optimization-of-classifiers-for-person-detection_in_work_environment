import sys
from pathlib import Path

# Add src/ directory to PYTHONPATH
SRC_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_PATH))
