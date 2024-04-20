import sys
import os
cur_path = os.path.abspath(__file__)
project_path = "/".join(cur_path.split("/")[:-3])
sys.path.insert(0, project_path)