import sys
from pathlib import Path

# Put the repo root on the path so tests can `import pipeline` when pytest is
# run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
