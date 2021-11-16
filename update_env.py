"""
Run to update env.yml file
"""
import os

os.system('conda env export -n qp_portfolio  > env.yml')
os.system('git commit -m "Updated env.yml" env.yml')
os.system('git push')