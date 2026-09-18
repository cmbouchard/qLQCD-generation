### Script to generate quenched Wilson gauge fields, with the option of tadpole- and Symanzik-improvement.
### Modified to include multiprocessing based on Panagiotis's script.
from __future__ import print_function
import argparse
import os
from gauge_latticeqcd import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate quenched Wilson gauge field configurations, with optional "
                     "rectangle (Symanzik) and/or tadpole improvement.")

    lat = parser.add_argument_group("lattice geometry")
    lat.add_argument("--Nt", type=int, default=6, help="temporal extent (default: 6)")
    lat.add_argument("--Nx", type=int, default=6, help="spatial extent, x (default: 6)")
    lat.add_argument("--Ny", type=int, default=6, help="spatial extent, y (default: 6)")
    lat.add_argument("--Nz", type=int, default=6, help="spatial extent, z (default: 6)")

    chain = parser.add_argument_group("Markov chain")
    chain.add_argument("--startcfg", type=int, default=0,
                        help="cold start (0) or existing cfg number to resume the Markov chain "
                             "from (default: 0)")
    chain.add_argument("--Ncfg", type=int, default=2002,
                        help="number of lattices to generate; add 2 to the number you actually "
                             "want (default: 2002)")
    chain.add_argument("--action", choices=["W", "WR", "W_T", "WR_T"], default="WR_T",
                        help="W = Wilson, WR = Wilson with rectangle improvement; append _T to "
                             "either for tadpole improvement (default: WR_T)")
    chain.add_argument("--beta", type=float, default=5.7, help="beta = 6/g^2 (default: 5.7)")
    chain.add_argument("--Nhits", type=int, default=10, help="hits between each update (default: 10)")
    chain.add_argument("--epsilon", type=float, default=0.3,
                        help="how far from identity each update matrix is; tune for a 20-50%% "
                             "acceptance ratio, e.g. for beta=5.7, 8^4: 0.2 -> 50%%, 0.25 -> 42%%, "
                             "0.3 -> 34%% (default: 0.3)")

    tad = parser.add_argument_group("tadpole improvement (ignored unless --action ends in _T)")
    tad.add_argument("--Nu0_step", type=int, default=1,
                      help="number of cfgs to skip between calculating u0 (default: 1)")
    tad.add_argument("--Nu0_avg", type=int, default=25,
                      help="number of u0 values to average together before updating (default: 25)")
    tad.add_argument("--u0", type=float, default=1.,
                      help="u0 = <W11>^(1/4); 1 for a cold start, or the value to continue an "
                           "existing lattice from (default: 1.0)")
    tad.add_argument("--freeze-u0", dest="freeze_u0", action="store_true",
                      help="once u0 has stabilised, pass this (with --u0 set to the stabilised "
                           "value, the same --action, and --startcfg continuing the chain) to "
                           "stop recalculating u0 for the production run")

    return parser.parse_args()


def main():
    args = parse_args()

    dir_name = (args.action + '_' + str(args.Nt) + 'x' + str(args.Nx) + 'x' + str(args.Ny) + 'x'
                + str(args.Nz) + '_b' + str(int(args.beta * 100)))

    ### create output directory if it does not exist
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        print("Directory exists for beta ", args.beta)

    generate(beta=args.beta, u0=args.u0, action=args.action, Nt=args.Nt, Nx=args.Nx, Ny=args.Ny,
             Nz=args.Nz, startcfg=args.startcfg, Ncfg=args.Ncfg, Nhits=args.Nhits,
             epsilon=args.epsilon, Nu0_step=args.Nu0_step, Nu0_avg=args.Nu0_avg,
             freeze_u0=args.freeze_u0)


if __name__ == "__main__":
    main()
