"""PySide6 GUI launcher for Manga Helper."""

import sys

try:
    from src.gui_pyqt import main
except Exception as exc:
    print("Unable to launch GUI:", exc, file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
