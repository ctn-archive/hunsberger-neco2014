"""
Compute the mutual information across a range of noise and heterogeneity values
"""

import time
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T

import metrics
import signals
import neurons


def info_sim(model, Nstds, bths, n_trials=10, alpha=True, save_name=None):
    Non = 32
    Noff = 32
    N = Non + Noff
    enc = np.concatenate([np.ones(Non), -np.ones(Noff)])

    ### signal
    Sstd = 0.1

    ### simulation parameters
    twarmup = 0.5
    ttrial = 4
    ttotal = ttrial + twarmup

    infos = np.nan * np.zeros((len(Nstds), len(bths), n_trials))
    spikes_sec = infos.copy()

    print ("Running info_sim: "
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
        R, spikes = neurons.simulate_theano(S, dt, enc=enc,
                                            bth=bth_vectors, Nstd=Nstd,
                                            tau_c=0.02,
                                            model=model, spikes_out=True)
        print "Done Nstd=%0.3e in t=%0.3f s" % (Nstd, time.time() - timer)

        ### take only valid trials and times
        Rvalid = np.isfinite(R).all(axis=-1)
        if not Rvalid.all():
            print "Some invalid trials (%d / %d)" % ((~Rvalid).sum(), Rvalid.size)

        t = dt * np.arange(S.shape[-1])
        tvalid = t > twarmup

        Sv = S[Rvalid][:,tvalid]
        Rv = R[Rvalid][:,tvalid]

        ### normalize
        Sv = metrics.normalize(Sv, axis=-1)
        Rv = metrics.normalize(Rv, axis=-1)

        ### compute metrics
        info = np.array([metrics.mutual_info(s, r) for s,r in zip(Sv,Rv)])
        infos[i,Rvalid] = info

        spike_sec = spikes[Rvalid][:,tvalid].sum(-1) / (ttrial * N)
        spikes_sec[i,Rvalid] = spike_sec

    if save_name is not None:
        np.savez(save_name, Nstds=Nstds, bths=bths,
                 infos=infos, spikes_sec=spikes_sec)
        print "Saved as %s" % save_name

    return infos, spikes_sec


def run(model, save_name=None):
    if save_name is None:
        save_name = "info_%s.npz" % model

    Nstds = 10**np.linspace(-4, 0, 41)
    bths = 10**np.linspace(-3, 0, 31)
    n_trials = 100
    info_sim(model, Nstds, bths, n_trials=n_trials,
             alpha=True, save_name=save_name)
