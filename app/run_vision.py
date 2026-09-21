"""
VISION Launcher
===============
Launches the VISION Streamlit application.
This is the entry point for the executable.
"""

import sys
import os
from pathlib import Path

# Add the current directory to path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

# Set Streamlit config path
os.environ['STREAMLIT_CONFIG_DIR'] = str(current_dir / '.streamlit')


def main():
    """Launch VISION application."""
    import streamlit.web.cli as stcli

    # Get the app.py path
    app_path = current_dir / 'app.py'

    if not app_path.exists():
        print(f"Error: app.py not found at {app_path}")
        sys.exit(1)

    # Launch Streamlit
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless", "true",
        "--server.port", "8501",
        "--server.maxUploadSize", "1024",
        "--server.maxMessageSize", "1024",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ]

    sys.exit(stcli.main())


if __name__ == "__main__":
    print("=" * 60)
    print("VISION - Versatile Intelligent Segmentation for")
    print("Image-based Observation of Nanoparticles")
    print("=" * 60)
    print("\nStarting VISION...")
    print("Open your browser to: http://localhost:8501")
    print("Press Ctrl+C to stop\n")

    main()
