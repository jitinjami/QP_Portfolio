"""
This file makes it more convenient to update the .yml file and keep the git up to date.
Run to update and push eca.yml file
"""
import os

os.system('conda env export -n qp_portfolio  > eca.yml')
os.system('git commit -m "Updated eca.yml" eca.yml')
os.system('git push')