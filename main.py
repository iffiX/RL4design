import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Make sure c++ modules are compiled
from renesis.sim import *
from navigator.main import navigator_main
from launch.run_experiment import run_experiment

if __name__ == "__main__":
    """
    1) Launch any experiment file in a temporary snapshot to prevent
       code from being changed while running.

            python main.py -r experiments/some_some_experiment/some_optimize.py

    2) Launch navigator for visualizing data
        If you have launched experiments before:

            python main.py -n

        Otherwise you may also specify a path to the result directory
        (i.e. the directory where ray results are saved)

            python main.py -n ~/ray_results
    """
    parser = argparse.ArgumentParser(
        description="Script for running experiments and visualizing results."
    )

    # Add the necessary arguments
    parser.add_argument("-r", "--run", metavar="PYTHON FILE", help="Run an experiment.")
    parser.add_argument(
        "-n",
        "--navigate",
        metavar="OPTIONAL RESULT PATH",
        help="Start navigator for visualizing results.",
        nargs="?",
        const="",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Check which argument was provided and execute the corresponding code
    if args.run:
        print(f"Running single experiment: {args.run}")
        run_experiment(args.run)

    elif args.navigate is not None:
        navigator_main(args.navigate if len(args.navigate) > 0 else None)
