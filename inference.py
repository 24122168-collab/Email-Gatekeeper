import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from app import main

if __name__ == "__main__":
    main()
