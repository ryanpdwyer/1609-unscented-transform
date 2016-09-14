# -*- coding: utf-8 -*-
"""
What is the unscented transform?
================================

Helper functions for a talk on the unscented transform.

"""
from __future__ import division, print_function
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import h5py
import ipywidgets as ipy
import uncertainties
from uncertainties import umath, ufloat
from matplotlib.patches import Ellipse
from matplotlib import colors


def svdsqrtm(x, eps=1e-15):
    """Return the matrix square root of x calculating using the svd.
    
    Set singular values < eps to 0.
    This prevents numerical errors from making sqrtm(x) complex
    (small eigenvalues of x accidently negative)."""
    u, s, v = linalg.svd(x)
    s_pos = np.zeros_like(s)
    for i in xrange(s.size):
        if s[i] > eps:
            s_pos[i] = s[i]

    return np.dot(u, np.dot(np.diag(np.sqrt(s_pos)), v.T))


def unscented(x, P, scale=1.0):
    """For a vector x of length N with covariance matrix P,
    form 2N sigma points used for taking the unscented transform.

                                      __
    Defaults to taking points at x ± √NP."""
    x = x.ravel() # Force shape
    N = x.size
    Psqrt = scale * svdsqrtm(N * P)
    x_sigma = []

    for i in xrange(N):
        x_sigma.append(x + Psqrt[:, i])

    for i in xrange(N):
        x_sigma.append(x - Psqrt[:, i])

    return x_sigma

def unscented_mu_cov(x_sigma):
    """Approximate mean, covariance from 2N sigma points transformed through
    an arbitrary non-linear transformation.

    Returns a flattened 1d array for x."""
    N = len(x_sigma)
    pts = np.array(x_sigma)

    x_mu = np.mean(pts, axis=0)
    diff = pts - x_mu
    x_cov = np.dot(diff.T, diff) / N
    return x_mu, x_cov

phis = np.linspace(0, 2*np.pi, 101)

uA = ufloat(1, 0.1)

def magphase(A, phi):
    return A * np.sin(phi)

def fill_donut(ax, radii, n=100, **kwargs):
    """Fill a donut between minimum radii and maximum radii.
    
    See http://stackoverflow.com/q/22789356"""
    phis = np.linspace(0, 2*np.pi, n)
    xs = np.outer(radii, np.cos(phis))
    ys = np.outer(radii, np.sin(phis))
    
    # in order to have a closed area, the circles
    # should be traversed in opposite directions
    xs[1,:] = xs[1,::-1]
    ys[1,:] = ys[1,::-1]
    
    return ax.fill(np.ravel(xs), np.ravel(ys), **kwargs)

def cov_ellipse(mu, cov, **kwargs):
    cov_vals, cov_vecs = linalg.eig(cov)
    
    return Ellipse(mu, 2*cov_vals[0].real**0.5, 2*cov_vals[1].real**0.5,
                      -np.arctan2(cov_vecs[0, 1], cov_vecs[0, 0])*180/np.pi,
                      **kwargs)


def magtime_phi_linear(mu_phi, sigma_A, sigma_phi):
    mu_x = magphase(1.0, mu_phi)
    cov = np.diag([sigma_A**2, sigma_phi**2])
    phi_x = phis[:51]
    fig, ax = plt.subplots(figsize=(8,8))
    axis_color='0.5'

    # Add axes
    ax.axhline(color=axis_color)
    ax.axvline(color=axis_color)
    
    # Mean circle (mu_A = 1)
    ax.plot(np.cos(phis), np.sin(phis))
    Amax = 1+sigma_A
    fill_donut(ax, [1-sigma_A, 1+sigma_A], edgecolor='none', alpha=0.1)
    
    
    ax.plot([0, np.cos(mu_phi)], [0, np.sin(mu_phi)], 'g')
    phi_plot = np.linspace(mu_phi+sigma_phi, mu_phi-sigma_phi, 9)
    ax.fill(np.r_[0, Amax*np.cos(phi_plot)], Amax*np.r_[0, np.sin(phi_plot)], 'g', alpha=0.1)
    ax.plot(np.cos(mu_phi), np.sin(mu_phi), 'ko', markeredgecolor='none')
    
    # Propogate error
    uA = ufloat(1.0, sigma_A)
    u_phi = ufloat(mu_phi, sigma_phi)
    ux = umath.cos(u_phi) * uA
    uy = umath.sin(u_phi) * uA
    ucov = np.array(uncertainties.covariance_matrix([ux, uy]))
    # Determine ellipse major and minor axes
    cov_vals, cov_vecs = linalg.eig(ucov)
    
    ellipse = Ellipse((ux.n, uy.n), 2*cov_vals[0].real**0.5, 2*cov_vals[1].real**0.5,
                      -np.arctan2(cov_vecs[0, 1], cov_vecs[0, 0])*180/np.pi,
                      color='c', alpha=0.5)
    ax.add_artist(ellipse)
    
    ax.set_xlim(-0.5, 1.25)
    ax.set_ylim(-0.5, 1.25)


def magtime_phi_exact(mu_phi, sigma_A, sigma_phi, xlim, ylim):
    mu_x = magphase(1.0, mu_phi)
    cov = np.diag([sigma_A**2, sigma_phi**2])
    phi_x = phis[:51]
    fig, ax = plt.subplots(figsize=(8,8))
    axis_color='0.5'
    
    Amax = 1+sigma_A

    # Add exact propogation of samples
    pts = 500
    A_pts = np.random.randn(pts)*sigma_A + 1
    phi_pts = np.random.randn(pts)*sigma_phi + mu_phi
    x_pts = A_pts * np.cos(phi_pts)
    y_pts = A_pts * np.sin(phi_pts)
    mu_xy = np.array([x_pts.mean(), y_pts.mean()])
    vec = np.c_[x_pts - x_pts.mean(), y_pts - y_pts.mean()]
    cov_pts = np.dot(vec.T, vec) / pts
    
    # Add axes
    ax.axhline(color=axis_color)
    ax.axvline(color=axis_color)
    
    # Mean circle (mu_A = 1)
    ax.plot(np.cos(phis), np.sin(phis))
    fill_donut(ax, [1-sigma_A, 1+sigma_A], edgecolor='none', alpha=0.1)
    
    
    ax.plot([0, np.cos(mu_phi)], [0, np.sin(mu_phi)], 'g')
    phi_plot = np.linspace(mu_phi+sigma_phi, mu_phi-sigma_phi, 9)
    ax.fill(np.r_[0, Amax*np.cos(phi_plot)], np.r_[0, Amax*np.sin(phi_plot)], 'g', alpha=0.1)
    ax.plot(np.cos(mu_phi), np.sin(mu_phi), color='c', marker='o', markeredgecolor='none')
    # KDE plot is the right idea, but too cluttered right now.
    #     sns.kdeplot(x_pts, y_pts, ax=ax, cmap=plt.cm.Blues)
    
    # Propogate error
    uA = ufloat(1.0, sigma_A)
    u_phi = ufloat(mu_phi, sigma_phi)
    ux = umath.cos(u_phi) * uA
    uy = umath.sin(u_phi) * uA
    ucov = np.array(uncertainties.covariance_matrix([ux, uy]))
    ax.add_artist(cov_ellipse((ux.n, uy.n), ucov, color='c', alpha=0.2, lw=2))
    pt_color = colors.cnames['navy']
    ax.scatter(x_pts, y_pts, c=pt_color, edgecolors='none', alpha=0.1)
    ax.scatter(*mu_xy, s=50, c=pt_color, edgecolors='none')
    ax.add_artist(cov_ellipse(mu_xy, cov_pts, edgecolor=pt_color, fill=False, lw=2.0))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    

def magtime_phi_unscented(mu_phi, sigma_A, sigma_phi, xlim, ylim):
    mu_x = magphase(1.0, mu_phi)
    cov = np.diag([sigma_A**2, sigma_phi**2])
    phi_x = phis[:51]
    fig, ax = plt.subplots(figsize=(8,8))
    axis_color='0.5'

    Amax = 1 + sigma_A
    
    # Add exact propogation of samples
    pts = 500
    A_pts = np.random.randn(pts)*sigma_A + 1
    phi_pts = np.random.randn(pts)*sigma_phi + mu_phi
    x_pts = A_pts * np.cos(phi_pts)
    y_pts = A_pts * np.sin(phi_pts)
    mu_xy = np.array([x_pts.mean(), y_pts.mean()])
    vec = np.c_[x_pts - x_pts.mean(), y_pts - y_pts.mean()]
    cov_pts = np.dot(vec.T, vec) / pts
    
    # Unscented propogation
    x = np.array([1.0, mu_phi])
    cov = np.diag([sigma_A**2, sigma_phi**2])
    X = unscented(x, cov)
    Y = [np.array([x[0]*np.cos(x[1]),
                   x[0]*np.sin(x[1])]) for x in X]

    mu_y, cov_y = unscented_mu_cov(Y)

    # Add axes
    ax.axhline(color=axis_color)
    ax.axvline(color=axis_color)
    
    # Mean circle (mu_A = 1)
    ax.plot(np.cos(phis), np.sin(phis))
    fill_donut(ax, [1-sigma_A, 1+sigma_A], edgecolor='none', alpha=0.1)
    
    
    ax.plot([0, np.cos(mu_phi)], [0, np.sin(mu_phi)], 'g')
    phi_plot = np.linspace(mu_phi+sigma_phi, mu_phi-sigma_phi, 9)
    ax.fill(np.r_[0, Amax*np.cos(phi_plot)], np.r_[0, Amax*np.sin(phi_plot)], 'g', alpha=0.1)
    ax.plot(np.cos(mu_phi), np.sin(mu_phi), color='c', marker='o', markeredgecolor='none')
    
    # Propogate error
    uA = ufloat(1.0, sigma_A)
    u_phi = ufloat(mu_phi, sigma_phi)
    ux = umath.cos(u_phi) * uA
    uy = umath.sin(u_phi) * uA
    ucov = np.array(uncertainties.covariance_matrix([ux, uy]))
    ax.add_artist(cov_ellipse((ux.n, uy.n), ucov, color='c', alpha=0.2, lw=2))
    pt_color = colors.cnames['navy']
    
    ax.scatter(x_pts, y_pts, c='0.6', edgecolors='none', alpha=0.15)
    ax.scatter(*mu_xy, s=50, c=pt_color, edgecolors='none')
    ax.add_artist(cov_ellipse(mu_xy, cov_pts, edgecolor=pt_color, fill=False, lw=2.0))
    
    Yarray = np.array(Y).T
    ax.scatter(Yarray[0], Yarray[1], c='m', s=35, edgecolors='none')
    ax.scatter(*mu_y, c='m', s=50, edgecolors='none')
    ax.add_artist(cov_ellipse(mu_y, cov_y, edgecolor='m', fill=False, lw=2.0))


    
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


x0 = np.array([1.0, 0.0, 1.0]).reshape((1, -1))

def step_time(F, x, P, Q):
    xm = np.dot(F, x)
    Pm = np.dot(np.dot(F, P), F.T) + Q
    return xm, Pm

def Hx(x):
    u"""For a state vector x (A, φ, ω), return derivatives
    dy/dx at the current state."""
    xx = x.ravel()
    return np.array([np.cos(xx[1]),
                    -xx[0] * np.sin(xx[1]),
                    0.0]).reshape((1, -1))

def Yfunc(x):
    """Evaluate position y for a state vector x (A, φ, ω)."""
    return x[0]*np.cos(x[1])


def measure_EKF(Yfunc, Hfunc, xm, Pm, R, y_meas):
    H = Hfunc(xm)

    y_est = float(Yfunc(xm))
    K = np.dot(Pm, H.T) / float(np.dot(np.dot(H, Pm), H.T) + R)
    err = float(y_meas - y_est)
    xp = xm + K * err

    one_minus_KH = (np.eye(xm.size) - np.dot(K, H))
    Pp = np.dot(np.dot(one_minus_KH, Pm), one_minus_KH.T) + R * np.dot(K, K.T)
    return y_est, xp, Pp


def measure_UKF(Yfunc, xm, Pm, R, y_meas):
    X = unscented(xm, Pm)
    Y = [Yfunc(x) for x in X]

    y_est, Py_no_r = unscented_mu_cov(Y)

    Py = float(Py_no_r + R)
    x_diff = np.array(X) - xm.ravel()
    y_diff = (np.array(Y) - y_est).reshape((-1, 1))

    Pxy = np.dot(x_diff.T, y_diff)


    err = float(y_meas - y_est) # 1x1

    # Now do the measurement update
    # Only works for single measurement!
    K = Pxy / Py
    xp = xm +  K * err
    Pp = Pm - np.dot(K, Py * K.T)
    return y_est, xp, Pp


class EKF(object):
    def __init__(self, F, Q, R, Yfunc, Hfunc, x, P):
        self.F = F
        self.Q = Q
        self.R = R
        self.Hf = Hfunc
        self.Yf = Yfunc
        self.M = F.shape[0]  # Dimension of state vector
        self.x = x
        self.P = P    
    
    def measure(self, y_meas):
        self.xm = self.x
        self.Pm = self.P

        y_est, xp, Pp = measure_EKF(self.Yf, self.Hf, self.x, self.P, self.R, y_meas)
    
        self.x = xp
        self.P = Pp

        return y_est, xp, Pp
        
    def step(self):
        self.xp = self.x
        self.Pp = self.P
        x, P = step_time(self.F, self.x, self.P, self.Q)
        self.x = x
        self.P = P
        
        return x, P

class UKF(object):
    def __init__(self,  F, Q, R, Yfunc, x, P):
        self.F = F
        self.Q = Q
        self.R = R
        self.Yf = Yfunc
        self.M = F.shape[0]  # Dimension of state vector
        self.x = x
        self.P = P
    
    def step(self):
        self.xp = self.x
        self.Pp = self.P
        x, P = step_time(self.F, self.x, self.P, self.Q)
        self.x = x
        self.P = P
        
        return x, P
    
    def measure(self, y_meas):
        self.xm = self.x
        self.Pm = self.P

        y_est, xp, Pp = measure_UKF(self.Yf, self.x, self.P, self.R, y_meas)
    
        self.x = xp
        self.P = Pp
        
        return y_est, xp, Pp
    
class Sim(object):
    def __init__(self, F, Q, R, Yfunc, x0, yield_x=True):
        self.F = F
        self.Q = Q
        self.R = R
        self.Yf = Yfunc
        self.Mx = x0.size  # Dimension of state vector
        self.x = x0
        self.yield_x = yield_x
        
        y = Yfunc(x0)
        self.My = 1
    
    # Python 3 compatibility
    def __next__(self):
        return self.next()
    
    def next(self):
        x_next = np.dot(self.F, self.x) + np.dot(self.Q, np.random.randn(self.Mx, 1))
        self.x = x_next
        y_next = float(Yfunc(x_next) + np.random.randn(self.My) * self.R)
        
        if self.yield_x:
            return x_next, y_next
        else:
            return y_next
        
def Hx(x):
    u"""For a state vector x (A, φ, ω), return derivatives
    dy/dx at the current state."""
    x = x.ravel()
    return np.array([
            np.cos(x[1]),
            -x[0] * np.sin(x[1]), 0.0]).reshape((1, -1))

def Yfunc(x):
    """Evaluate position y for a state vector x (A, φ, ω)."""
    x = x.ravel()
    return float(x[0]*np.cos(x[1]))


