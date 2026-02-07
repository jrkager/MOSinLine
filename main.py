# ------------------
# Import codebase of RLRP and PATT
# ------------------
from rlrp import RLRP
from patt import PATT

# Other imports
import argparse

# ------------------
# Define data structures
# ------------------
class Instance:
    pass

class RLRPResult:
    # warehouse choice and size
    # association of customers to warehouses per scenario
    pass

class PATTResult:
    # delivery pttern per store
    # routes per day
    pass


if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.parse_args()
