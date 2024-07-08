#!/usr/bin/env python3

from __future__ import division
import argparse, os, time


import pandas as pd
import numpy as np
import numpy.matlib
import scipy.stats
from colorama import Fore, Style


pd.options.mode.chained_assignment = None

from mrp_functions import *

def print_banner():

    """
    Prints ASCII Art Banner + Author Info.

    """

    banner_string = f"""{Fore.RED}
    __  __ ____  ____
    |  \\/  |  _ \\|  _ \\
    | |\\/| | |_) | |_) |
    | |  | |  _ <|  __/
    |_|  |_|_| \\_\\_|{Style.RESET_ALL}

    {Fore.GREEN}Production Author:{Style.RESET_ALL}
    Guhan Ram Venkataraman, Ph.D.

    {Fore.GREEN}Contact:{Style.RESET_ALL}
    Email: guhan@stanford.edu
    URL: https://github.com/rivas-lab/mrp

    {Fore.GREEN}Methods Developers:{Style.RESET_ALL}
    Manuel A. Rivas, Ph.D.; Matti Pirinen, Ph.D.
    Rivas Lab | Stanford University

    {Fore.GREEN}References:{Style.RESET_ALL}
    Venkataraman, et al. Am J Hum Genet. (2021).
    https://doi.org/10.1016/j.ajhg.2021.11.005
    """

    print(banner_string)


def return_input_args(args):

    """
    Further parses the command-line input.

    Makes all lists unique; calculates S and K; and creates lists of appropriate
        matrices.

    Parameters:
    args: Command-line arguments that have been parsed by the parser.

    Returns:
    df: Merged summary statistics.
    map_file: Input file containing summary statistic paths + pop and pheno data.
    S: Number of populations/studies.
    K: Number of phenotypes.
    pops: Unique set of populations (studies) to use for analysis.
    phenos: Unique set of phenotypes to use for analysis.
    R_study: Unique list of R_study matrices to use for analysis.

    """

    try:
        map_file = pd.read_csv(
            args.map_file,
            sep="\t",
            dtype={"path": str, "pop": str, "pheno": str, "R_phen": bool},
        )
    except:
        raise IOError("File specified in --file does not exist.")

    df, pops, phenos, S, K = read_in_summary_stats(
        map_file, args.metadata_path, args.exclude, args.sigma_m_types, args.build, args.chrom
    )
    for arg in vars(args):
        if (arg != "filter_ld_indep") and (arg != "mean"):
            setattr(args, arg, sorted(list(set(getattr(args, arg)))))
    R_study = [
        np.diag(np.ones(S)) if x == "independent" else np.ones((S, S))
        for x in args.R_study_models
    ]
    return (df, map_file, S, K, pops, phenos, R_study)


def range_limited_float_type(arg):

    """
    Type function for argparse - a float within some predefined bounds.

    Parameters:
    arg: Putative float.

    Returns:
    f: The same float, if a valid floating point number between 0 and 1.

    """

    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("must be valid floating point numbers.")
    if f <= 0 or f > 1:
        raise argparse.ArgumentTypeError("must be > 0 and <= 1.")
    return f


def positive_float_type(arg):

    """
    Type function for argparse - a float that is positive.

    Parameters:
    arg: Putative float.

    Returns:
    f: The same float, if positive.

    """

    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("must be valid floating point numbers.")
    if f < 0:
        raise argparse.ArgumentTypeError("must be >= 0.")
    return f


def initialize_parser():

    """
    Parses inputs using argparse.

    """

    parser = argparse.ArgumentParser(
        description="MRP takes in several variables that affect how it runs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        dest="map_file",
        help="""path to tab-separated file containing list of:
         summary statistic file paths,
         corresponding studies,
         phenotypes, and
         whether or not to use the file in R_phen generation.

         format:

         path        study        pheno        R_phen
         /path/to/file1   study1    pheno1     TRUE
         /path/to/file2   study2    pheno1     FALSE
         """,
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        dest="metadata_path",
        help="""path to tab-separated file containing:
         variants,
         gene symbols,
         consequences,
         MAFs,
         and LD independence info.

         format:

         V       gene_symbol     most_severe_consequence maf  ld_indep
         1:69081:G:C     OR4F5   5_prime_UTR_variant     0.000189471     False
        """,
    )
    parser.add_argument(
        "--build",
        choices=["hg19", "hg38"],
        type=str,
        required=True,
        dest="build",
        help="""genome build (hg19 or hg38). Required.""",
    )
    parser.add_argument(
        "--chrom",
        choices=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","X","Y"],
        type=str,
        nargs="+",
        dest="chrom",
        default=[],
        help="""chromosome filter. options include 1-22, X, and Y""",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs=1,
        dest="mean",
        default=0,
        help="""prior mean of genetic effects (Default: 0).""",
    )
    parser.add_argument(
        "--R_study",
        choices=["independent", "similar"],
        type=str,
        nargs="+",
        default=["similar"],
        dest="R_study_models",
        help="""type of model across studies.
         options: independent, similar (default: similar). can run both.""",
    )
    parser.add_argument(
        "--R_var",
        choices=["independent", "similar"],
        type=str,
        nargs="+",
        default=["independent"],
        dest="R_var_models",
        help="""type(s) of model across variants.
         options: independent, similar (default: independent). can run both.""",
    )
    parser.add_argument(
        "--M",
        choices=["variant", "gene"],
        type=str,
        nargs="+",
        default=["gene"],
        dest="agg",
        help="""unit(s) of aggregation.
         options: variant, gene (default: gene). can run both.""",
    )
    parser.add_argument(
        "--sigma_m_types",
        choices=["sigma_m_mpc_pli", "sigma_m_var", "sigma_m_1", "sigma_m_005"],
        type=str,
        nargs="+",
        default=["sigma_m_mpc_pli"],
        dest="sigma_m_types",
        help="""scaling factor(s) for variants.
         options: var (i.e. 0.2 for ptvs, 0.05 for pavs/pcvs),
         1, 0.05 (default: mpc_pli). can run multiple.""",
    )
    parser.add_argument(
        "--variants",
        choices=["pcv", "pav", "ptv", "all"],
        type=str,
        nargs="+",
        default=["ptv"],
        dest="variant_filters",
        help="""variant set(s) to consider.
         options: proximal coding [pcv],
                  protein-altering [pav],
                  protein truncating [ptv],
                  all variants [all]
                  (default: ptv). can run multiple.""",
    )
    parser.add_argument(
        "--maf_thresh",
        type=range_limited_float_type,
        nargs="+",
        default=[0.01],
        dest="maf_threshes",
        help="""which MAF threshold(s) to use. must be valid floats between 0 and 1
         (default: 0.01).""",
    )
    parser.add_argument(
        "--se_thresh",
        type=positive_float_type,
        nargs="+",
        default=[0.2],
        dest="se_threshes",
        help="""which SE threshold(s) to use. must be valid floats between 0 and 1
         (default: 0.2). NOTE: This strict default threshold is best suited for binary
         summary statistics. For quantitative traits, we suggest the use of a higher
         threshold.""",
    )
    parser.add_argument(
        "--prior_odds",
        type=range_limited_float_type,
        nargs="+",
        default=[0.0005],
        dest="prior_odds_list",
        help="""which prior odds (can be multiple) to use in calculating posterior
         probabilities. must be valid floats between 0 and 1 (default: 0.0005, expect
         1 in 2000 genes to be a discovery).""",
    )
    parser.add_argument(
        "--p_value",
        choices=["farebrother", "davies", "imhof"],
        type=str,
        nargs="+",
        default=[],
        dest="p_value_methods",
        help="""which method(s) to use to convert Bayes Factors to p-values. if command
         line argument is invoked but method is not specified, will throw an error
         (i.e., specify a method when it is invoked). if not invoked, p-values will not
         be calculated. options: farebrother, davies, imhof. NOTE: --p_value imports R
         objects and methods, which slows down MRP. farebrother is fastest and
         recommended if p-values are a must.""",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs=1,
        default=[],
        dest="exclude",
        help="""path to file containing list of variants to exclude from analysis.

         format of file:

         1:69081:G:C
         1:70001:G:A
        """,
    )
    parser.add_argument(
        "--filter_ld_indep",
        action="store_true",
        dest="filter_ld_indep",
        help="""whether or not only ld-independent variants should be kept (default: False;
         i.e., use everything).""",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        nargs=1,
        default=[],
        dest="out_folder",
        help="""folder to which output(s) will be written (default: current folder).
         if folder does not exist, it will be created.""",
    )
    parser.add_argument(
        "--out_filename",
        type=str,
        nargs=1,
        default=[],
        dest="out_filename",
        help="""file prefix with which output(s) will be written (default: underscore-delimited
         phenotypes).""",
    )
    return parser


def mrp_main():
    """
    Runs MRP analysis on summary statistics with the parameters specified
        by the command line.

    """

    parser = initialize_parser()
    args = parser.parse_args()
    print_banner()
    print("")
    print("Valid command line arguments. Importing required packages...")
    print("")

    if args.p_value_methods:
        print("")
        print(
            Fore.RED
            + "WARNING: Command line arguments indicate p-value generation. "
            + "This can cause slowdowns of up to 12x."
        )
        print(
            "Consider using --prior_odds instead to generate posterior probabilities"
            + " as opposed to p-values."
            + Style.RESET_ALL
        )
        import rpy2
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri

        rpy2.robjects.numpy2ri.activate()
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects.vectors import StrVector
        from rpy2.robjects.vectors import ListVector
        from rpy2.robjects.vectors import FloatVector
        import warnings
        from rpy2.rinterface import RRuntimeWarning

        warnings.filterwarnings("ignore", category=RRuntimeWarning)

    out_folder = args.out_folder[0] if args.out_folder else os.getcwd()
    out_filename = args.out_filename[0] if args.out_filename else []
    df, map_file, S, K, pops, phenos, R_study_list = return_input_args(args)
    if args.filter_ld_indep:
        df = df.loc[df["ld_indep"], ]
    for se_thresh in args.se_threshes:
        se_df = se_filter(df, se_thresh, pops, phenos)
        err_corr, R_phen = return_err_and_R_phen(
            se_df, pops, phenos, len(pops), len(phenos), map_file
        )
        print("Correlation of errors, SE threshold = " + str(se_thresh) + ":")
        print(err_corr)
        print("")
        print("R_phen:")
        print(R_phen)
        print("")
        loop_through_parameters(
            se_df,
            se_thresh,
            args.maf_threshes,
            args.agg,
            args.variant_filters,
            S,
            R_study_list,
            args.R_study_models,
            pops,
            K,
            R_phen,
            phenos,
            args.R_var_models,
            args.sigma_m_types,
            err_corr,
            args.prior_odds_list,
            args.p_value_methods,
            out_folder,
            out_filename,
            args.mean,
            args.chrom,
        )


if __name__ == "__main__":
    start = time.time()
    mrp_main()
    end = time.time()
    print(
        Fore.CYAN +
        'MRP analysis finished. Total elapsed time: {:.2f} [s]'.format(
            end - start
        ) + Style.RESET_ALL
    )
