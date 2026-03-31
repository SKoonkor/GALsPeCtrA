import argparse
from pathlib import Path

from galspectra.sampling.paramgrid import generate_lhs_grid
from galspectra.sps.fsps_backend import create_stellar_population
from galspectra.sed.generator import generate_seds
from galspectra.sed.io import save_sed_grid


def main():
    parser = argparse.ArgumentParser(
            prog="galspectra",
            description="GALsPeCtrA: SED generation pipeline"
            )

    subparsers = parser.add_subparsers(dest="command")

    # generate command
    gen = subparsers.add_parser("generate", help="Generate SED grid")

    gen.add_argument("--n-samples", type=int, default=100,
                     help="Number of samples in parameter grid")

    gen.add_argument("--logzsol", type=float, default=0.0,
                     help="Metallicity (log Z/Zsun)")

    gen.add_argument("--output", type=str, default="data/sed_grid.npz",
                     help="Output file path relative to project root")


    args = parser.parse_args()

    if args.command == "generate":
        run_generate(args)
    else:
        parser.print_help()



def run_generate(args):

    print ("Test run")


