import jax
import jax.numpy as jnp
from functools import partial

def init_ma_components(H_ma):
    H_ma = jnp.atleast_1d(H_ma)
    m = len(H_ma.ravel())
    F_ma = jnp.diagflat(jnp.ones(m-1), k=-1)
    T_m = jnp.zeros(m).at[0].set(1)[:, None]
    return H_ma, F_ma, T_m


def init_ar_components(H_ar):
    H_ar = jnp.atleast_1d(H_ar)
    r = len(H_ar.ravel())
    F_ar = jnp.diagflat(jnp.ones(r-1), k=-1)
    F_ar = F_ar.at[0].set(H_ar)
    T_r = jnp.zeros(r).at[0].set(1)[:, None]
    return H_ar, F_ar, T_r


def build_arma_components(H_ar, H_ma):
    H_ar, F_ar, T_ar = init_ar_components(H_ar)
    H_ma, F_ma, T_ma = init_ma_components(H_ma)

    r, m = len(T_ar), len(T_ma)
    F_arma = jnp.zeros((r + m, r + m))
    F_arma = F_arma.at[:r,:r].set(F_ar)
    F_arma = F_arma.at[r:, r:].set(F_ma)

    # H_arma = jnp.concat([H_ar, H_ma])
    H_arma = jnp.r_[H_ar, H_ma]

    T_arma = jnp.r_[T_ar, T_ma]
    return H_arma, F_arma, T_arma


def init_arma_components(H_ar=None, H_ma=None):
    """
    Build H, F, and T matrices for an ARMA process.
    If one of H_ar or H_ma is None,
    we build the arma process only with the specified component.
    """
    if (H_ar is None) and (H_ma is not None):
        H, F, T = init_ma_components(H_ma)
    elif (H_ar is not None) and (H_ma is None):
        H, F, T = init_ar_components(H_ar)
    elif (H_ar is not None) and (H_ma is not None):
        H, F, T = build_arma_components(H_ar, H_ma)
    else:
        raise KeyError("One of H_ar and H_ma must be specified")

    return H, F, T


def step_ssm(state, key, F, H, T, R):
    theta, noise = state
    
    # Build next state
    theta = F @ theta + T @ noise
    noise = jax.random.normal(key) * jnp.sqrt(R)
    
    # Build observation 
    y = H @ theta + noise

    out = {
        "y": y,
        "theta": theta
    }

    state_next = (theta, noise)
    return state_next, out


def sample_ssm(key, F, H, T, R, n_steps):
    """
    Run sample of a state-space model
    with initial zero-valued latent process
    """
    dim_latent = F.shape[0]
    theta_init = jnp.zeros((dim_latent, 1))
    noise_init = jnp.eye(1) * 0.0
    state_init = (theta_init, noise_init) # latent and error term

    keys = jax.random.split(key, n_steps)
    step = partial(step_ssm, F=F, H=H, T=T, R=R)
    _, hist = jax.lax.scan(step, state_init, keys)
    hist = jax.tree.map(jnp.squeeze, hist)
    return hist