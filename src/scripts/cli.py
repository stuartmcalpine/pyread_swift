import os
import argparse
from .grid_snapshots import grid_particles
from .combine import combine

# Parse input parameters.
parser = argparse.ArgumentParser(
    description="The pyread_swift CLI interface",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
subparsers = parser.add_subparsers(title="subcommand", dest="subcommand")

# Add a select region grid to a SWIFT snapshot.
arg_grid = subparsers.add_parser(
    "grid", help="Add a select region grid to a SWIFT snapshot"
)

arg_grid.add_argument("input", help="Path to snapshot", type=str)
arg_grid.add_argument("output", help="Path to desired output snapshot", type=str)
arg_grid.add_argument(
    "--region_min", help="Min coordinates for grid (only for zooms)", type=float
)
arg_grid.add_argument(
    "--region_max", help="Max coordinates for grid (only for zooms)", type=float
)

# Combine multiple snapshot parts into a single SWIFT snapshot.
arg_combine = subparsers.add_parser(
    "combine", help="Combine multiple SWIFT snapshot parts into one (DM only)"
)
arg_combine.add_argument("input", help="Path to snapshot part", type=str)
arg_combine.add_argument(
    "output", help="Path to desired output combined snapshot", type=str
)


def main():
    args = parser.parse_args()

    if args.subcommand == "grid":
        grid_particles(args.input, args.output, args.region_min, args.region_max)

    elif args.subcommand == "combine":
        combine(args.input, args.output)
