import subprocess
import sys

if sys.platform == 'darwin':
    subprocess.run(['xhost', '+'])
