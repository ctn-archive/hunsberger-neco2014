"""
Measure how noise and heterogeneity affect the phase, and therefore the
synchronization, of neurons.
"""

import time
import numpy as np
import numpy.random as npr

import signals
import neurons


def phase_sim(model, Nstds, bths, n_trials=10, alpha=True, save_name=None):
    N = 64
    enc = np.ones(N)

    ### signal
    Sstd = 0.1

    ### simulation parameters
    twarmup = 0.5
    ttrial = 4
    ttotal = ttrial + twarmup

    phases = np.nan * np.zeros((len(Nstds), len(bths), n_trials))

    print ("Running phase_sim: "
           "model = %s, n_Nstds = %d, n_bths = %d, n_trials = %d, N = %d") % (
        model, len(Nstds), len(bths), n_trials, N)
    for i, Nstd in enumerate(Nstds):
        dt = 0.1e-3

        n_bths = len(bths)
        bth_vectors = np.array(
            [[npr.uniform(low=-bth, high=bth, size=N) for k in xrange(n_trials)]
             for bth in bths])

        n_signals = n_bths * n_trials
        if not alpha:
            Sfreq = 5
            Sbandwidth = Sfreq
            S = signals.equalpower(dt, ttotal, max_freq=Sfreq, std=Sstd, n=n_signals)
        else:
            Stau = 20e-3
            Sbandwidth = 1. / (2 * np.pi * Stau)
            S = signals.alpha_white(dt, ttotal, tau=Stau, std=Sstd, n=n_signals)
        S = S.reshape((n_bths, n_trials, -1))

        timer = time.time()
        [phase] = neurons.simulate_theano(S, dt, enc=enc,
                                          bth=bth_vectors, Nstd=Nstd,
                                          tau_c=0.02, model=model,
                                          resp_out=False, phase_out=True)
        print "Done Nstd=%0.3e in t=%0.3f s" % (Nstd, time.time() - timer)
        phases[i] = phase.mean(-1)

    if save_name is not None:
        np.savez(save_name, Nstds=Nstds, bths=bths, phases=phases)
        print "Saved as %s" % save_name

    return phases


def run(model, save_name=None):
    if save_name is None:
        save_name = "phase_%s.npz" % model

    Nstds = 10**np.linspace(-4, 0, 41)
    bths = 10e-3 * np.array([0, 1, 5, 20])
    n_trials = 100
    phase_sim(model, Nstds, bths,
              n_trials=n_trials, alpha=True, save_name=save_name)
