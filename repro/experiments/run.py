"""Fixed run command dispatcher for the orx reproduction tree.

The project run command is identical on every node; which claim to run is
encoded in committed code (repro/experiments/TARGET) so each node's branch
selects its experiment by editing that one file.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))


def main():
    target_file = os.path.join(HERE, "TARGET")
    if not os.path.exists(target_file):
        sys.exit("repro/experiments/TARGET missing: choose claims12 | claim3 | claim4")
    target = open(target_file).read().strip()
    if target == "claims12":
        from repro.experiments import claims12
        claims12.run_all()
    elif target == "claim3":
        from repro.experiments import claim3
        claim3.run_all()
    elif target == "claim4":
        from repro.experiments import claim4
        claim4.run_all()
    else:
        sys.exit(f"unknown TARGET: {target}")


if __name__ == "__main__":
    main()
