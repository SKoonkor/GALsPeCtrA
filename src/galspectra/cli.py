import argparse
from pathlib import Path

from galspectra.sampling.paramgrid import generate_lhs_grid
from galspectra.sps.fsps_backend import create_stellar_population
from galspectra.sed.generator import generate_seds
from galspectra.sed.io import save_sed_grid
from galspectra.config import load_config


def main():
    parser = argparse.ArgumentParser(
            prog="galspectra",
            description="GALsPeCtrA: SED generation pipeline"
            )

    subparsers = parser.add_subparsers(dest="command")

    # generate command
    gen = subparsers.add_parser("generate", help="Generate SED grid")

    gen.add_argument("--config", type=str, help="Path to YAML config file")

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

    # Resolve project root
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    if args.config:
        config = load_config(args.config)
        print(f"load config: {args.config}")
    else:
        raise ValueError("You must provide --config")


    pg = config["paramgrid"]

    params = pg["parameters"]

    if pg["type"] == "lhs":
        param_dict = generate_lhs_grid(
                params,
                n_samples=pg.get("n_samples", 100)
                )
    else:
        raise NotImplementedError("Only LHS currently supported")

    print (f"Generated {len(param_dict['samples'])} samples")

    # FSPS
    fsps_cfg = config.get("fsps", {})

    sp = create_stellar_population(logzsol=args.logzsol)
    print ("FSPS initialised")

    # Generate SEDs
    sed_data = generate_seds(param_dict, sp)

    # Save SEDs
    OUTPUT_FILE = config["output"]["file"]
    OUTPUT_PATH = PROJECT_ROOT/OUTPUT_FILE

    save_sed_grid(OUTPUT_PATH, sed_data, config=config)

    print (f"\nSaved SEDs to: {OUTPUT_PATH}")


