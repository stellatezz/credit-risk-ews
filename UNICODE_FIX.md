# Unicode Encoding Fix - Phase 2 Update

## Problem
**Error:** `UnicodeEncodeError: 'charmap' codec can't encode character '\u2713' in position 2`

The problematic character was: **✓** (Unicode U+2713 - checkmark)

This error occurs when Python tries to print Unicode characters to Windows console, which defaults to the 'charmap' codec (Windows-1252), unable to encode special Unicode symbols.

## Root Cause
Multiple Python source files contained Unicode characters in `print()` statements that are executed at console runtime.

These characters get encoded using the system console codec, which on Windows defaults to charmap.

## Solution
Replaced all non-ASCII Unicode characters in console `print()` statements with ASCII-safe alternatives:

| Unicode | Code Point | Replacement | Files |
|---------|-----------|-------------|-------|
| ✓ | U+2713 | [OK] | loaders.py |
| ✗ | U+2717 | [FAIL] | loaders.py |
| → | U+2192 | -> | loaders.py |
| — | U+2014 | -- | loaders.py, pipeline.py |
| ≥ | U+2265 | >= | labels.py |
| ± | U+00B1 | +/- | eval.py |

## Files Modified
- `src/ews/loaders.py` - 7 replacements (checkmarks, crosses, arrow, em-dashes)
- `src/ews/pipeline.py` - 1 replacement (em-dash in title)
- `src/ews/labels.py` - 1 replacement (greater-or-equal symbol)
- `src/ews/eval.py` - 1 replacement (plus-minus symbol)

## Testing
To verify the fix works, run:
```powershell
python src/run.py
```

You should now see output like:
```
Loading daily prices (yfinance, with data/raw/ cache)...
  [OK] AAL: 3654 days (cached)
  [OK] BBBY: 3654 days (2010-01-04 -> 2024-05-30)
  [OK] F: 3654 days (cached)
```

Instead of the previous Unicode error.

## Streamlit App
The Streamlit UI files (`project_home.py`, `pages/*.py`) contain emoji characters (🏢, 📊, ✅, ⚠️, etc.), but these are safe because:
1. They're rendered in the Streamlit web interface, not printed to console
2. Streamlit handles UTF-8 encoding internally
3. No UnicodeEncodeError will occur from these

## Phase 2 Data Integration
You can now proceed with:
1. Running the pipeline with phase 2 firms (80 companies)
2. Downloading SEC data and yfinance prices
3. Regenerating Streamlit outputs
4. No more encoding errors on Windows console


