"""
Run to update eca.yml file
"""
import os

os.system('conda env export -n qp_portfolio  > eca.yml')
os.system('git commit -m "Updated eca.yml" eca.yml')
os.system('git push')