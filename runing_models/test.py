import os
import sys


print("path1:", sys.path)

sys.path.append(".")

print("path2:", sys.path)


print("cwd:", os.getcwd())
