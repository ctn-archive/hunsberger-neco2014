"""
ODE simulation of the neuron models in Theano
"""

import theano
import theano.tensor as T
import theano.sandbox.rng_mrg
import collections

import numpy as np
import numpy.random as npr

def simulate_theano(S, dt, enc, bth, Nstd, x0=None, tau_c=0.01, model='fhn',
                    resp_out=True, states_out=False,
                    spikes_out=False, phase_out=False):

    dtype = theano.config.floatX
    enc = enc.astype(dtype)

    trials_shape = S.shape[0:-1]
    nt = S.shape[-1]    # last axis holds number of time points
    Ntrials = reduce(lambda x,y:x*y, trials_shape)
    N = len(enc)
    NN = (Ntrials, N)

    S = S.reshape(Ntrials, nt)
    if bth.ndim > 2: bth = bth.reshape(Ntrials, N)
    else:            assert bth.shape == (Ntrials, N)

    ### define model-specific simulation code (ODEs, etc.)
    if model == 'fhn':
        b = -bth + 0.3216
        dt_sim = 1000*dt # convert dt to milliseconds, since FHN equations are in ms
        t_buf = 1

        if x0 is None:
            v0 = npr.uniform(low=-2, high=2, size=NN)
            w0 = npr.uniform(low=-0.4, high=1.2, size=NN)
        else:
            v0, w0 = x0

        v = theano.shared(v0.astype(dtype), name='v')
        w = theano.shared(w0.astype(dtype), name='w')
        r = theano.shared(t_buf*np.ones(NN, dtype=dtype), name='r')
        states = [v, w, r]

        b = T.cast(b, dtype=dtype)
        dt1 = T.cast(dt_sim, dtype=dtype)
        dt2 = T.cast(np.sqrt(dt_sim), dtype=dtype)

        def get_states_y(u, eta, states):
            v, w, r = states
            vnew = v + dt1*(v - (v**3)/3 - w + b + u) + dt2*eta
            wnew = w + dt1*(0.08*(v + 0.7 - 0.8*w))

            rnew = T.switch(vnew > 0.0, r + dt1, r - dt1)
            rnew = rnew.clip(-t_buf, t_buf)
            spikes = (rnew >= 0) & (r < 0)
            unspikes = (rnew <= 0) & (r > 0)
            rnew = T.switch(spikes, 1000, rnew)
            rnew = T.switch(unspikes, -1000, rnew)
            y = T.dot(100*spikes, enc/enc.size)
            return [vnew, wnew, rnew], y

        def get_spikes(states):
            v, w, r = states
            return (r > 999).sum(-1)

        def get_phase(states):
            v, w, r = states

            ### convert to centered complex coordinates
            Vcenter = -0.22
            Wcenter = 0.6
            x = v - Vcenter
            y = w - Wcenter
            angle = T.arctan2(y, x)

            ### take the mean of unit vectors
            mag = T.sqrt(x**2 + y**2)
            x = x / mag
            y = y / mag
            mean = T.arctan2(y.mean(-1), x.mean(-1))

            ### calculate angles around the mean
            angle = T.mod(angle - mean[:,None] + np.pi, 2*np.pi) - np.pi
            std = T.sqrt((angle**2).mean(-1))
            return std

    elif model == 'lif':

        Vspike = 100
        Vth = 1
        Vth1 = (Vth + Vspike) / 2.0

        alpha = 15
        tauRC = 20e-3
        tRef = 33e-3
        iRef = np.round(tRef / dt)

        b = Vth - alpha*bth
        b = T.cast(b, dtype=dtype)
        dt1 = T.cast(dt, dtype=dtype)
        dt2 = T.cast(np.sqrt(dt), dtype=dtype)
        iRef = T.cast(iRef, dtype=dtype)

        if x0 is None:
            v0 = npr.uniform(low=0, high=Vth, size=NN)
        else:
            v0 = x0[0]
            assert v0.shape == NN
        w0 = np.zeros(NN)

        v = theano.shared(v0.astype(dtype), name='v')
        w = theano.shared(w0.astype(dtype), name='w')
        states = [v, w]

        def get_states_y(u, eta, states):
            v, w = states
            v1 = v + dt1/tauRC*(alpha*u + b - v) + dt2/tauRC*alpha*eta
            v1 = T.switch(w > 0.5, v1, 0)
            vnew = T.switch(v1 > Vth, Vspike, v1)
            wnew = T.switch(v1 > Vth, -iRef, w + 1)
            y = T.dot(T.switch(vnew > Vth1, vnew, 0), enc/enc.size)
            return [vnew, wnew], y

        def get_spikes(states):
            v, w = states
            return (v > Vth1).sum(-1)

        def get_phase(states):
            v, w = states
            angle = T.switch(w > 0,
                             np.pi * v.clip(0, 1),
                             w * (np.pi / T.abs_(T.min(w))))

            mean = T.arctan2(T.sin(angle).mean(axis=-1),
                             T.cos(angle).mean(axis=-1))

            ### calculate angles around the mean
            angle = T.mod(angle + (np.pi - mean[:,None]), 2*np.pi) - np.pi
            std = T.sqrt((angle**2).mean(-1))
            return std
    else:
        raise ValueError("Unrecognized model type '%s'" % model)

    ### create Theano function to move forward a single time step
    tau_filter = T.cast(dt / (tau_c + dt), dtype=dtype) # backward-Euler method
    def get_updates(u, eta, states, r):
        u_n = u[:,None] * enc[None,:]
        newstates, y = get_states_y(u_n, eta, states)
        rnew = r + tau_filter*(y - r)
        return newstates, rnew

    nx = len(states)
    u = T.vector(name='u')
    eta = T.matrix(name='eta')
    r = theano.shared(np.zeros(Ntrials).astype(dtype), name='r')

    newstates, rnew = get_updates(u, eta, states, r)
    updates = collections.OrderedDict(zip(states, newstates))
    updates[r] = rnew

    spikes = get_spikes(newstates)
    phase = get_phase(newstates)

    t_outs = []
    if resp_out:   t_outs.append(rnew)
    if states_out: t_outs.append(T.as_tensor(newstates))
    if spikes_out: t_outs.append(spikes)
    if phase_out:  t_outs.append(phase)

    f_step = theano.function([u, eta], t_outs, updates=updates,
                             allow_input_downcast=True)

    ### create arrays to hold the desired outputs
    outs = []
    outs_shapes = []
    if resp_out:
        outs.append(np.zeros((nt, Ntrials)))
        outs_shapes.append((nt,) + trials_shape)
    if states_out:
        outs.append(np.zeros((nt, nx, Ntrials, N)))
        outs_shapes.append((nt, nx) + trials_shape + (N,))
    if spikes_out:
        outs.append(np.zeros((nt, Ntrials)))
        outs_shapes.append((nt,) + trials_shape)
    if phase_out:
        outs.append(np.zeros((nt, Ntrials)))
        outs_shapes.append((nt,) + trials_shape)

    ### run the simulation
    S = S.T
    for i in xrange(nt):
        eta = npr.normal(size=NN, scale=Nstd)
        outsi = f_step(S[i], eta)
        for j, outij in enumerate(outsi):
            outs[j][i] = outij

    for i, out in enumerate(outs):
        ### reshape to target shape
        out = out.reshape(outs_shapes[i])

        ### move first (time) axis to end
        axes = np.arange(out.ndim) + 1
        axes[-1] = 0
        outs[i] = np.transpose(out, axes=axes)

    return outs
