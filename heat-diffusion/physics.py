import numpy as np
import jax.numpy as jnp
import jax.random as jr
import jax


def get_square_map(shape):
    """
    Helper function for simulation of the heat equation
    Parameters
    ----------
    shape : 2D shape of the simulation state
    Returns
    -------

    """
    map_ = np.zeros(shape=shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            map_[i, j] = np.sqrt((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)
    map_ = jnp.array(map_)
    square_map = -jnp.square(map_)
    return square_map


def forward_full(initial_state, dt, t1, noise=0.0, rng=jr.PRNGKey(0)):
    """
    Forward simulation of the heat equation from initial_state at time 0 until time t1 with time step dt.
    Parameters
    ----------
    initial_state : jnp.array
    dt : float
    t1 : float
    noise : float
    rng : jr.PRNGKey

    Returns jnp.array
    -------

    """
    steps = int(t1 / dt)
    shape = initial_state.shape[-2:]

    square_map = get_square_map(shape)
    scale_factor = jnp.exp(square_map * dt)

    @jax.jit
    def physics_forward(state_):
        field_fft = jnp.fft.fft2(jnp.array(state_))
        field_shiftfft = jnp.fft.fftshift(field_fft, axes=[-1, -2])
        field_shiftfft = jnp.array(field_shiftfft)

        field_updated = scale_factor * field_shiftfft
        field_difference = field_updated - field_shiftfft

        drift = jnp.real(jnp.fft.ifft2(jnp.fft.ifftshift(field_difference, axes=[-1, -2])))

        return drift

    states = [initial_state]
    state = initial_state
    for _ in range(steps):
        state = state + physics_forward(state)
        state = state + noise * jr.normal(rng, shape=state.shape)
        rng, _ = jr.split(rng)
        states.append(state)

    return states


def get_forward_full_custom(shape):
    shape = shape[-2:]

    square_map = get_square_map(shape)

    @jax.jit
    def forward_full_custom(initial_state, times, rng=jr.PRNGKey(0), noise=0.0):

        states = [initial_state]
        state = initial_state
        for i in range(times.shape[0] - 1):
            t1 = times[i + 1]
            t0 = times[i]

            field_fft = jnp.fft.fft2(jnp.array(state))
            field_shiftfft = jnp.fft.fftshift(field_fft, axes=[-1, -2])
            field_shiftfft = jnp.array(field_shiftfft)

            scale_factor = jnp.exp(square_map * (t1 - t0))
            field_updated = scale_factor * field_shiftfft
            field_difference = field_updated - field_shiftfft

            drift = jnp.real(jnp.fft.ifft2(jnp.fft.ifftshift(field_difference, axes=[-1, -2])))

            state = state + drift
            state = state + jnp.sqrt(t1 - t0) * noise * jr.normal(rng, shape=state.shape)
            rng, _ = jr.split(rng)
            states.append(state)

        return states

    return forward_full_custom


def forward_step(shape, DT):
    shape = shape[-2:]

    square_map = get_square_map(shape)
    scale_factor = jnp.exp(square_map * DT)

    def physics_forward(state_):

        field_fft = jnp.fft.fft2(jnp.array(state_))
        field_shiftfft = jnp.fft.fftshift(field_fft, axes=[-1, -2])
        field_shiftfft = jnp.array(field_shiftfft)

        field_updated = scale_factor * field_shiftfft
        field_difference = field_updated - field_shiftfft

        drift = jnp.real(jnp.fft.ifft2(jnp.fft.ifftshift(field_difference, axes=[-1, -2])))

        return drift

    return physics_forward


def forward_step_custom(shape):
    shape = shape[-2:]
    map_ = np.zeros(shape=shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            map_[i, j] = jnp.sqrt((i - shape[0] / 2) ** 2 + (j - shape[1] / 2) ** 2)

    map_ = jnp.array(map_)

    square_map = -jnp.square(map_)

    def physics_forward(state_, t):

        field_fft = jnp.fft.fft2(jnp.array(state_))
        field_shiftfft = jnp.fft.fftshift(field_fft, axes=[-1, -2])
        field_shiftfft = jnp.array(field_shiftfft)

        scale_factor = jnp.exp(square_map * t)
        field_updated = scale_factor * field_shiftfft
        field_difference = field_updated - field_shiftfft

        drift = jnp.real(jnp.fft.ifft2(jnp.fft.ifftshift(field_difference, axes=[-1, -2])))

        return drift

    return physics_forward
