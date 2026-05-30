"""Pipeline entry point.

Wrapper that lets you run the EWS pipeline from the repo root without
setting PYTHONPATH:

    python src/run.py

All the actual logic lives in src/ews/*.py; this file just adds src/ to
sys.path and delegates to ews.pipeline.main().
"""

import os
import sys
import io

# Force UTF-8 encoding for Windows console (cp1252 by default)
# This prevents UnicodeEncodeError when printing special characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ews.pipeline import main

if __name__ == "__main__":
    main()
