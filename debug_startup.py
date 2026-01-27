import sys
import traceback
import os

try:
    print("Attempting to import app from src.main...")
    sys.path.append(os.getcwd())
    from src.main import app
    print("Import successful!")
    print("App object:", app)
except Exception as e:
    print("IMPORT FAILED!")
    traceback.print_exc()
    with open("debug_startup.log", "w") as f:
        traceback.print_exc(file=f)
