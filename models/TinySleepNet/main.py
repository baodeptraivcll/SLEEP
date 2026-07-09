import sys
import os

# Add root project folder to python path to resolve common package and root main.py imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from main import main as root_main

if __name__ == "__main__":
    if "--architecture" not in sys.argv:
        sys.argv.extend(["--architecture", "TinySleepNet"])
    root_main()
