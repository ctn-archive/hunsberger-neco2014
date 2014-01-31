"""
Generate tuning curves for neurons under various levels of noise.
"""

import time
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T

import signals
import neurons


def tuning_sim(model, Nstds, Svals, n_trials=3, save_name=None):
    N = 30
    enc = np.ones(N)
    bth_vectors = np.zeros((len(Svals), n_trials, N))

    ### signal
    dt = 0.1e-3
    Sstd = 0.1

    ### simulation parameters
    twarmup = 0.5
    ttrial = 4
    ttotal = ttrial + twarmup

    nt = int(ttotal / dt)
    t = dt * np.arange(nt)
    tvalid = t >= twarmup

    rates = np.zeros((len(Nstds), len(Svals), n_trials))

    print ("Running tuning_sim: "
           "model = %s, n_Nstds = %d, n_Svals = %d, trials = %d, N = %d") % (
        model, len(Nstds), len(Svals), n_trials, N)
    for i, Nstd in enumerate(Nstds):
        S = np.array(
            [[sval * np.ones(nt) for j in xrange(n_trials)] for sval in Svals])

        timer = time.time()
        [spikes] = neurons.simulate_theano(S, dt, enc=enc,
                                           bth=bth_vectors, Nstd=Nstd,
                                           tau_c=0.02, model=model,
                                           resp_out=0, spikes_out=1)
        print "Done Nstd=%0.3e in t=%0.3f s" % (Nstd, time.time() - timer)

        rates[i] = spikes[:,:,tvalid].sum(-1) / float(N * ttrial)

    if save_name is not None:
        np.savez(save_name,
                 Nstds=Nstds, Svals=Svals, rates=rates)
        print "Saved as %s" % save_name

    return rates


def run(model, save_name=None):
    if save_name is None:
        save_name = "info_%s.npz" % model

    Svals = np.linspace(-0.2, 0.2, 101)
    if model == 'lif':
        Nstds = np.array([1e-9, 3e-3, 1e-2, 3e-2])
    elif model == 'fhn':
        Nstds = np.array([1e-9, 3e-2, 1e-1, 3e-1])
    n_trials = 5
    tuning_sim(model, Nstds, Svals, n_trials=5, save_name=save_name)
