"""
Generate firing information at a given noise and heterogeneity,
for spike rasters
"""

import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T

import signals
import neurons


def firing_sim(model, Nstds, bths, save_name=None):
    Non = 32
    Noff = 32
    N = Non + Noff
    Ntrials = 1

    enc = np.concatenate([np.ones(Non), -np.ones(Noff)])

    dt = 0.1e-3

    t_warmup = 0.5
    t_trial = 1
    t_total = t_trial + t_warmup
    t = np.linspace(dt, t_total, np.round(t_total/dt))
    nt = len(t)

    ### signal
    Sstd = 0.1
    Sfreq = 5
    Stau = 20e-3
    S = signals.alpha_white(dt, t_total, tau=Stau, std=Sstd, n=Ntrials)

    ### simulations
    results = []
    for Nstd in Nstds:
        resultsi = []
        for bth in bths:
            bth_vals = npr.uniform(low=-bth, high=bth, size=(Ntrials, N))
            R, X = neurons.simulate_theano(S, dt, enc, bth_vals, Nstd,
                                           model=model, states_out=True)
            v = X[0]

            assert np.isfinite(R).all() # ensure all trials are finite

            ### shuffle order of neurons
            vi = v[0].copy()
            npr.shuffle(vi)

            ### take only simulation times after warmup
            tmask = t > t_warmup
            tvalid = t[tmask] - t_warmup
            vvalid = vi[:,tmask]
            spiketimes = [tvalid[vv > 2] for vv in vvalid]

            resultsi.append(spiketimes)
        results.append(resultsi)

    if save_name is not None:
        np.savez(save_name, noisehetero=results, Nstds=Nstds, bths=bths,
                 t=t, S=S)
        print "Saved as: %s" % save_name

    return results


def run(model, save_name=None):
    if save_name is None:
        save_name = "firing_%s.npz" % model

    Nstds = np.array([1e-3, 1e-2, 1e-1])
    bths = np.array([1e-2, 1e-1, 1e-0])
    firing_sim(model, Nstds, bths, save_name=save_name)
