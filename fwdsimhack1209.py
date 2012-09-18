"""
This is a hack.
"""

import argparse
import time
import random
from collections import deque
from itertools import product, izip_longest

import numpy as np


# wait this many generations before attempting to get the ancestral lineage
BUILDUP_DELAY = 1000

# enumerate the haplotype states
g_AB = 0
g_Ab = 1
g_aB = 2
g_ab = 3

def _recombination_helper():
    r = {}
    for a, b in product(range(4), repeat=2):
        r[a, b] = ((a&2)|(b&1), (b&2)|(a&1))
    return r

def _mutation_helper():
    m = {}
    for a in range(4):
        m[a] = (a^2, a^1)
    return m

g_recombine = _recombination_helper()
g_mutate = _mutation_helper()

class TimeoutException(Exception): pass
class IterationException(Exception): pass
class FinishedException(Exception): pass


class Generation:
    def __init__(self):
        self.generation_index = None
        self.ps1 = None # array of parents of site 1
        self.ps2 = None # array of parents of site 2
        self.hap = None # array of haplotypes

class GenerationSummary:
    @classmethod
    def get_header_string(cls):
        headers = (
                'pop.AB', 'pop.Ab', 'pop.aB', 'pop.ab',
                'ancestral.site.1', 'ancestral.site.2',
                'ancestral.haploid.collision',
                )
        return '\t'.join(headers)
    def __init__(self):
        self.generation_index = None
        self.pop_AB = None
        self.pop_Ab = None
        self.pop_aB = None
        self.pop_ab = None
        self.ancestral_site_1 = None
        self.ancestral_site_2 = None
        self.ancestral_haploid_collision = None
    def __str__(self):
        d = {g_AB : '"AB"', g_Ab : '"Ab"', g_aB : '"aB"', g_ab : '"ab"'}
        r = {True : 'T', False : 'F'}
        return '\t'.join(str(x) for x in (
            self.generation_index,
            self.pop_AB,
            self.pop_Ab,
            self.pop_aB,
            self.pop_ab,
            d[self.ancestral_site_1],
            d[self.ancestral_site_2],
            r[self.ancestral_haploid_collision],
            ))


def grouper(n, iterable, fillvalue=None):
    # this is an official itertools recipe
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


###############################################################################
# Put these types into their own module
# if this script is ever broken up.

def positive_integer(s):
    x = int(s)
    if x <= 0:
        raise argparse.ArgumentTypeError(
                '%r is not a positive integer' % s)
    return x

def nonnegative_integer(s):
    x = int(s)
    if x < 0:
        raise argparse.ArgumentTypeError(
                '%r is not a non-negative integer' % s)
    return x

def positive_float(s):
    x = float(s)
    if x <= 0:
        raise argparse.ArgumentTypeError(
                '%r is not a positive floating point number' % s)
    return x

def nonnegative_float(s):
    x = float(s)
    if x < 0:
        raise argparse.ArgumentTypeError(
                '%r is not a non-negative floating point number' % s)
    return x

class restricted_float:
    # this is unused...
    def __init__(self,
            low_inclusive=None, low_exclusive=None, 
            high_inclusive=None, high_exclusive=None):
        e = Exception('internal float restriction conflict')
        if low_inclusive is not None and low_exclusive is not None:
            raise e
        if high_inclusive is not None and high_exclusive is not None:
            raise e
        low = low_inclusive
        if low_exclusive is not None:
            low = low_exclusive
        high = high_inclusive
        if high_exclusive is not None:
            high = high_exclusive
        if high is not None and low is not None:
            if high > low:
                raise e
        if low_exclusive is not None and high_exclusive is not None:
            if low_exclusive == high_exclusive:
                raise e
        self.low_inclusive = low_inclusive
        self.low_exclusive = low_exclusive
        self.high_inclusive = high_inclusive
        self.high_exclusive = high_exclusive
    def __call__(self, s):
        x = float(s)
        if self.low_inclusive is not None and not (x >= low_inclusive):
            raise argparse.ArgumentTypeError(
                    '%r is not >= %r' % (s, low_exclusive))
        if self.low_exclusive is not None and not (x > low_exclusive):
            raise argparse.ArgumentTypeError(
                    '%r is not > %r' % (s, low_exclusive))
        if self.high_inclusive is not None and not (x <= high_inclusive):
            raise argparse.ArgumentTypeError(
                    '%r is not <= %r' % (s, high_exclusive))
        if self.high_exclusive is not None and not (x < high_exclusive):
            raise argparse.ArgumentTypeError(
                    '%r is not < %r' % (s, high_exclusive))
        return x



###############################################################################
# This could possibly go into its own module,
# for running things until hitting an iteration cap or time cap
# or user break.

def run(fn, args=(), max_iterations=None, max_seconds=None):
    try:
        iter_count = 0
        if max_seconds is not None:
            tm = tm_last = time.time()
        while True:
            if max_iterations is not None:
                if iter_count >= max_iterations:
                    raise IterationException
                    #return
            if max_seconds is not None:
                if tm - tm_last >= max_seconds:
                    raise TimeoutException
                    #return
            if not fn(*args):
                raise FinishedException
                #return
            iter_count += 1
            if max_seconds is not None:
                tm_last = tm
                tm = time.time()
    except KeyboardInterrupt as e:
        return


###############################################################################

class Sim:
    def __init__(self, N_hap, m_rate, r_rate, fitnesses, burn_in, max_samples):
        # store the simulation settings
        self.N_hap = N_hap
        self.m_rate = m_rate
        self.r_rate = r_rate
        self.fitnesses = fitnesses
        self.burn_in = burn_in
        self.max_samples = max_samples
        # initialize some internal state
        self._next_generation_index = 0
        self.q = deque()
        self.nburned = 0
        self.nsampled = 0
        self.buildup = 0
    def add_generation(self, g):
        g.generation_index = self._next_generation_index
        self._next_generation_index += 1
        self.q.append(g)
    def _evolve(self):
        """
        Paste a single generation onto the end of the queue.
        """
        # Init a temporary intra-generational state
        # which results from mutation and recombination but not selection.
        gtmp = Generation()
        # Set each parent index to its own position.
        gtmp.ps1 = np.arange(self.N_hap)
        gtmp.ps2 = np.arange(self.N_hap)
        # Copy the haplotypes from the parent generation.
        gtmp.hap = self.q[-1].hap.copy()
        # Mutate the haplotypes in place selecting haploids without repacement.
        nevents = np.random.poisson(self.m_rate)
        if nevents:
            if nevents > self.N_hap:
                raise Exception(
                        'mutation rate is too high -- '
                        'tried to sample too many things without replacement')
            indices = random.sample(range(self.N_hap), nevents)
            sites = np.random.randint(2, size=nevents)
            for index, site in zip(indices, sites):
                gtmp.hap[index] = g_mutate[gtmp.hap[index]][site]
        # Recombine in place selecting pairs without replacement.
        nevents = np.random.poisson(self.r_rate)
        if nevents:
            if nevents*2 > self.N_hap:
                raise Exception(
                        'recombination rate is too high -- '
                        'tried to sample too many things without replacement')
            indices = random.sample(range(self.N_hap), nevents*2)
            for a, b in grouper(2, indices):
                gtmp.ps2[a], gtmp.ps2[b] = gtmp.ps2[b], gtmp.ps2[a]
                gtmp.hap[a], gtmp.hap[b] = g_recombine[gtmp.hap[a], gtmp.hap[b]]
        # Get the multinomial probabilities.
        p = np.array([self.fitnesses[h] for h in gtmp.hap])
        p /= np.sum(p)
        # Get the multinomial sample.
        counts = np.random.multinomial(self.N_hap, p)
        # Init the new state.
        gnew = Generation()
        gnew.hap = np.empty(self.N_hap, dtype=int)
        gnew.ps1 = np.empty(self.N_hap, dtype=int)
        gnew.ps2 = np.empty(self.N_hap, dtype=int)
        # Iterate over parent and child indices.
        ci = 0
        for pi, nselected in enumerate(counts):
            for i in range(nselected):
                gnew.hap[ci] = gtmp.hap[pi]
                gnew.ps1[ci] = gtmp.ps1[pi]
                gnew.ps2[ci] = gtmp.ps2[pi]
                ci += 1
        # Append the new generation info to the deque.
        self.add_generation(gnew)
    def _backtrace(self):
        """
        This is called periodically.
        It may delete generations from the left end of the deque.
        @return: list of extracted generation summaries from recent to ancient
        """
        summaries = []
        # Init the search for the ancestral lineage.
        # Track the set of indices of ancestral haploids in each generation.
        # When the set has length 1 then this is the ancestral lineage.
        s1_set = set(range(self.N_hap))
        s2_set = set(range(self.N_hap))
        for i in reversed(range(1, len(self.q))):
            s1_set = set(self.q[i].ps1[x] for x in s1_set)
            s2_set = set(self.q[i].ps2[x] for x in s2_set)
            if len(s1_set) > 1 or len(s2_set) > 1:
                continue
            # At this point we have discovered the ancestral lineage
            # for both genomic sites.
            g = self.q[i-1]
            summary = GenerationSummary()
            summary.generation_index = g.generation_index
            summary.pop_AB = sum(1 for x in g.hap if x == 0)
            summary.pop_Ab = sum(1 for x in g.hap if x == 1)
            summary.pop_aB = sum(1 for x in g.hap if x == 2)
            summary.pop_ab = sum(1 for x in g.hap if x == 3)
            summary.ancestral_site_1 = g.hap[list(s1_set)[0]]
            summary.ancestral_site_2 = g.hap[list(s2_set)[0]]
            summary.ancestral_haploid_collision = (s1_set == s2_set)
            summaries.append(summary)
        for s in summaries:
            self.q.popleft()
        return summaries
    def __call__(self):
        """
        Do a single generation.
        Return False to stop the simulation.
        """
        # base case
        if not self.q:
            g = Generation()
            g.ps1 = np.zeros(self.N_hap, dtype=int)
            g.ps2 = np.zeros(self.N_hap, dtype=int)
            g.hap = np.zeros(self.N_hap, dtype=int)
            self.add_generation(g)
            return True
        # non-base case
        self._evolve()
        if self.nburned < self.burn_in:
            self.q.popleft()
            self.nburned += 1
        else:
            self.buildup += 1
        if self.buildup == BUILDUP_DELAY:
            self.buildup = 0
            for summary in reversed(self._backtrace()):
                print summary
                self.nsampled += 1
                if self.max_samples is not None:
                    if self.nsampled >= self.max_samples:
                        return False
        return True


def main(args):
    N_hap = args.haploid_population_size
    s = args.scaled_selection / N_hap
    fitnesses = [1.0, 1.0 - s, 1.0 - s, 1.0]
    sim = Sim(
            N_hap,
            args.mutation_rate,
            args.recombination_rate,
            fitnesses,
            args.burn_in,
            args.max_samples,
            )
    print GenerationSummary.get_header_string()
    run(sim, max_iterations=None, max_seconds=args.max_seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample '
            'the ancestral lineages at each site of a two-site genome '
            'in a population evolving according to Wright-Fisher dynamics '
            'under the effects of recombination, mutation, '
            'selection, and drift.  '
            'Specify a max number of generations or wall clock time '
            'on the command line, or stop the simulation '
            'using Ctrl-C.  '
            'Redirect stdout to write to a file.  '
            )
    parser.add_argument(
            '--haploid_population_size',
            type=positive_integer,
            default=200,
            help='haploid population size N '
            '(default: %(default)s)',
            )
    parser.add_argument(
            '--mutation_rate',
            type=positive_float,
            default=0.001,
            help='expected mutations per generation in the population '
            '(default: %(default)s)',
            )
    parser.add_argument(
            '--recombination_rate',
            type=nonnegative_float,
            default=0.001,
            help='expected recombinations per generation in the population '
            '(default: %(default)s)',
            )
    parser.add_argument(
            '--scaled_selection',
            type=nonnegative_float,
            default=0.1,
            help='N*s where 1-s is relative fitness of an unfit haplotype '
            '(default: %(default)s)',
            )
    parser.add_argument(
            '--burn_in',
            type=nonnegative_integer,
            default=100,
            help='run for this many generations before reporting stats '
            '(default: %(default)s)',
            )
    parser.add_argument(
            '--max_samples',
            type=nonnegative_integer,
            default=None,
            help='report stats for at most this many generations',
            )
    parser.add_argument(
            '--max_seconds',
            type=positive_integer,
            default=None,
            help='give up after this many seconds',
            )
    main(parser.parse_args())

