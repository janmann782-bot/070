"""Default hosting entrypoint. Same launcher as python run.py."""
import sys
sys.dont_write_bytecode = True
from run import main

if __name__ == '__main__':
    main()
