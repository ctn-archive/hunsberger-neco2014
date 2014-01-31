
import numpy as np
import numpy.random as npr

def equalpower(dt, t_final, max_freq, mean=0.0, std=1.0, n=None):
    """Generate a random signal with equal power below a maximum frequency

    Parameters
    ----------
    dt : float
        Time difference between consecutive signal points [in seconds]
    t_final : float
        Length of the signal [in seconds]
    max_freq : float
        Maximum frequency of the signal [in Hertz]
    mean : float
        Signal mean (default = 0.0)
    std : float
        Signal standard deviation (default = 1.0)
    n : integer
        Number of signals to generate

    Returns
    -------
    s : array_like
        Generated signal(s), where each row is a signal, and each column a time
    """

    vector_out = n is None
    n = 1 if n is None else n

    df = 1. / t_final    # fundamental frequency

    nt = np.round(t_final / dt)        # number of time points / frequencies
    nf = np.round(max_freq / df)      # number of non-zero frequencies
    assert nf < nt

    theta = 2*np.pi*npr.rand(n, nf)
    B = np.cos(theta) + 1.0j * np.sin(theta)

    A = np.zeros((n, nt), dtype=np.complex)
    A[:,1:nf+1] = B
    A[:,-nf:] = np.conj(B)[:,::-1]

    S = np.fft.ifft(A, axis=1).real

    S = (std / S.std(axis=1))[:,None] * (S - S.mean(axis=1)[:,None] + mean)
    if vector_out: return S.flatten()
    else:          return S


def lowpass_white(dt, t_final, tau, mean=0.0, std=1.0, n=None):
    """White noise filtered with a basic low-pass filter.

    The low-pass filter is given by f(t) = 1 / tau * exp(-t / tau)
    """
    vector_out = n is None
    n = 1 if n is None else n

    ### make white noise
    nt = int(np.round(t_final / dt))
    x = np.random.normal(size=(n, nt), scale=std*np.sqrt(2*tau/dt))

    ### filter white noise
    x0 = np.random.normal(size=n, scale=std)

    # alpha = np.exp(-dt / tau)
    alpha = dt / tau
    for i in range(nt):
        # s = alpha*s + (1 - alpha)*S[:,i]
        x0 += alpha*(x[:,i] - x0)
        x[:,i] = x0

    if mean != 0:
        x += mean

    if vector_out: return x.flatten()
    else:          return x

def alpha_white(dt, t_final, tau, mean=0.0, std=1.0, n=None):
    """White noise filtered with the alpha function.

    The alpha function is given by alpha(t) = (t / tau) * exp(-t / tau)
    """
    vector_out = n is None
    n = 1 if n is None else n

    ### make white noise
    nt = int(np.round(t_final / dt))

    v = dt / tau
    e2v = np.exp(2*v)
    sigma = std / (v * dt) / np.sqrt(e2v*(e2v + 1)/(e2v - 1)**3)
    # u = np.random.normal(size=(n, nt+2), scale=sigma)
    u = np.random.normal(size=(nt+2, n), scale=sigma)

    ### filter white noise
    # x = np.zeros((n, nt))
    x = np.zeros((nt, n))
    x0 = np.random.normal(size=n, scale=std)
    x1 = x0.copy()

    alpha = 1 + 2*tau/dt
    beta = 1 - 2*tau/dt
    ba = beta/alpha
    for i in range(nt):
        # x[:,i] = (-2*ba*x1 - ba**2*x0 +
        #            tau/alpha**2 * (u[:,i+2] + 2*u[:,i+1] + u[:,i]))
        # x1, x0 = x[:,i], x1
        x[i,:] = (-2*ba*x1 - ba**2*x0 +
                   tau/alpha**2 * (u[i+2,:] + 2*u[i+1,:] + u[i,:]))
        x1, x0 = x[i,:], x1

    # if mean != 0:
    #     x += mean

    ### fully normalize
    # x *= (std / x.std(-1))[...,None]
    # x += (mean - x.mean(-1))[...,None]
    x *= (std / x.std(0))
    x += (mean - x.mean(0))
    x = x.T

    if vector_out: return x.flatten()
    else:          return x

def lowpass_white_conv(dt, t_final, tau, mean=0.0, std=1.0, n=None):
    vector_out = n is None
    n = 1 if n is None else n

    ### make white noise
    nt = int(np.round(t_final / dt))
    S = np.random.normal(size=(n, nt), scale=std*np.sqrt(2*tau/dt))
    # S = np.random.normal(size=(n, nt), scale=std*2*np.sqrt(tau/dt))

    ### filter white noise
    t = dt * np.arange(np.round(5*tau / dt))
    e = (dt / tau) * np.exp(-t / tau)

    # t = dt * np.arange(np.round(7*tau / dt))
    # e = t * (dt / tau) * np.exp(-t / tau)
    # e /= e.sum()
    # print e.sum()

    for i in range(len(S)):
        S[i] = np.convolve(S[i], e, mode='full')[:nt]

    if vector_out: return S.flatten()
    else:          return S

def test_lowpass_white():
    import matplotlib.pyplot as plt
    # from util.timing import tic, toc

    dt = 1e-3
    t_final = 10.0
    tau = 10e-3
    # s = lowpass_white(dt, t_final, tau, n=100)
    # s = lowpass_white2(dt, t_final, tau, std=1.0, n=100)
    s = alpha_white(dt, t_final, tau, n=1000)
    t = dt * np.arange(s.shape[1])
    f = (1. / t_final) * np.arange(s.shape[1])
    print "mean", s.mean(), s.mean(1).std()
    print "std", s.std(1).mean()
    print "std0", s[:,t < 0.01].std(0).mean(), "std1", s[:,t > 5].std(0).mean()

    F = np.fft.fft(s, axis=1)
    Am = np.abs(F).mean(0)
    y = np.fft.ifft(F * F.conj(), axis=1)
    ym = y.mean(0)

    plt.figure()
    r, c = 1, 3
    plt.subplot(r,c,1)
    plt.plot(t[t < 1], s[0][t < 1])
    plt.subplot(r,c,2)
    plt.plot(t[t < 5*tau], ym[t < 5*tau] / ym[0])
    plt.subplot(r,c,3)
    nyq = 0.5/dt
    plt.semilogx(f[f < nyq], Am[f < nyq] / Am[1])
    plt.show()

def test_alpha_white():
    from util.timing import tic, toc

    dt = 0.1e-3
    t_final = 5.0
    tau = 20e-3
    tic()
    s = alpha_white(dt, t_final, tau, mean=0.05, std=0.1, n=1000)
    toc()
    print "mean", s.mean(), s.mean(1).std()
    print "std", s.std(1).mean(), s.std(1).std()
    # print "std0", s[:,t < 0.01].std(0).mean(), "std1", s[:,t > 5].std(0).mean()


if __name__ == '__main__':
    # test_lowpass_white()
    test_alpha_white()
