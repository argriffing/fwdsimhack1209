"""
This is a hack.
"""

import argparse


def positive_integer(x):
    x = int(x)
    if x < 1:
        raise TypeError('expected a positive integer')
    return x

def nonnegative_float(x):
    x = float(x)
    if x <= 0:
        raise TypeError('expected a non-negative floating point number')
    return x

def positive_float(x):
    x = float(x)
    if x < 0:
        raise TypeError('expected a positive floating point number')
    return x

def main(args):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--haploid_population_size',
            type=positive_integer,
            default=100,
            help='haploid population size N',
            )
    parser.add_argument(
            '--mutation_rate',
            type=positive_float,
            default=0.001,
            help='expected mutations per generation in the population',
            )
    parser.add_argument(
            '--recombination_rate',
            type=nonnegative_float,
            default=0.001,
            help='expected recombinations per generation in the population',
            )
    parser.add_argument(
            '--scaled_selection',
            type=nonnegative_float,
            default=0.1,
            help='N*s where 1-s is relative fitness of an unfit haplotype',
            )
    main(parser.parse_args())

