import os
import sys

tests = os.path.dirname(__file__)
PACKAGE_NAME = '../darov'
sys.path.insert(0, os.path.abspath(os.path.join(tests, PACKAGE_NAME)))
