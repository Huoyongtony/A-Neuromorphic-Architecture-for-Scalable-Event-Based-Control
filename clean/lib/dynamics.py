"""Dynamics helpers extracted from the original notebooks.

The bulk of this module mirrors the legacy ``Rebound_WTA_lib`` so that the
clean notebooks can reuse the exact same numerical behaviour. Additional
helpers (the ``ss_Luka*`` and ``simulation_*`` functions) live at the bottom of
the file – they were previously defined inside notebooks and are pulled out
here for a single, reusable source of truth.
"""

import math
import random as rd

import matplotlib.pyplot as plt
import numba
import numpy as np

@numba.njit
def Sigmoid(x,sc,sh):
    return 1/(1+np.exp((-x+sh)*sc))
    
def Syn2(vs, gain,res,sharpe=2):
    return gain / ( 1 + np.exp(-sharpe*(vs-res)) )

def Syn_hh(vs, gain,res):
    return gain / ( 1 + np.exp(-(vs-res)*1.5))

def f_ion(x):
    return -3*np.exp(-(x-3)**2)

def f2_ion(x):
    return -7*np.exp(-(x-5)**2/4)

def u_ion(x):
    return x**2

def u2_ion(x,max_gain=100):
    return min(x**2,max_gain)

def s_ion(x):
    return -2*np.exp(-(x-3)**2)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def Leaky_Relu(x,a=1,b=0):
    return a*x * (x > 0)+ b*x * (x < 0)


@numba.njit
def Syn(vs, gain, res, alpha=10):
    return gain / (1 + np.exp(-(vs - res) * alpha))

@numba.njit
def Syn_2(vs):
    return 1 / (1 + np.exp(-vs))


@numba.njit
def Relu(x):
    # Using np.maximum which is supported in nopython mode.
    return np.maximum(x, 0)

# @numba.njit
# def sodium(x):
#     # x can be a scalar or an array.
#     return Relu(x + 1) * 20 + (Relu(x + 3) - Relu(x + 1)) * 1 + (Relu(x + 5) - Relu(x + 3)) * 0.5

# @numba.njit
# def potassium(x):
#     return Relu(x + 1) * 70 + (Relu(x + 3) - Relu(x + 1)) * 4 + (Relu(x + 5) - Relu(x + 3)) * 0.5

@numba.njit
def ss_Relu(x, u):

    def sodium(x):
    # x can be a scalar or an array.
        return Relu(x + 1) * 20 + (Relu(x + 3) - Relu(x + 1)) * 1 + (Relu(x + 5) - Relu(x + 3)) * 0.0 + (Relu(-x-3)) * 20
    
    
    def potassium(x):
        return Relu(x + 1) * 70 + (Relu(x + 3) - Relu(x + 1)) * 4 + (Relu(x + 5) - Relu(x + 3)) * 0.0

    # Assume x and u are 2D arrays.
    dx = np.zeros(x.shape)
    n = x.shape[0]
    
    # Compute dx[:,1] and dx[:,2] with a loop.
    
    dx[:, 1] = -x[:, 1] + x[:, 0]
    dx[:, 2] = (-x[:, 2] + x[:, 0]) / 20.0

    # Compute leak, fast positive and slow negative components in a vectorized way.
    leak = 2 * (x[:, 0] + 1)
    fast_positive = sodium(x[:, 1])
    slow_negative = potassium(x[:, 2])
    
    
    dx[:, 0] = -leak[:] - fast_positive[:] * (x[:, 0] - 5) - slow_negative[:] * (x[:, 0] + 3) + u[:, 0]
    
    return dx



@numba.njit
def ss_Relu_slow_spike(x, u, height=1.5,time=5):

    def sodium(x):
    # x can be a scalar or an array.
        return Relu(x + 1) * 2*height + (Relu(x + 3) - Relu(x + 1)) * 1 + (Relu(x + 5) - Relu(x + 3)) * 0.0 + (Relu(-x-3)) * 20

    def potassium(x):
        return Relu(x + 1) * 7*height + (Relu(x + 3) - Relu(x + 1)) * 4 + (Relu(x + 5) - Relu(x + 3)) * 0.0 

    # Assume x and u are 2D arrays.
    dx = np.zeros(x.shape)
    n = x.shape[0]
    
    # Compute dx[:,1] and dx[:,2] with a loop.
    dx[:, 1] = (-x[:, 1] + x[:, 0])
    dx[:, 2] = (-x[:, 2] + x[:, 0]) / 20.0/time

    # Compute leak, fast positive and slow negative components in a vectorized way.
    leak = 2 * (x[:, 0] + 1)
    fast_positive = sodium(x[:, 1])
    slow_negative = potassium(x[:, 2])
    
    
    dx[:, 0] = (-leak[:] - fast_positive[:] * (x[:, 0] - 5) - slow_negative[:] * (x[:, 0] + 3) + u[:, 0])
    
    return dx



@numba.njit
def ss_Relu_para(x, u, para):

    def sodium(x,b_1,b_2,m_1,m_2):
    # x can be a scalar or an array.
        return Relu(x + b_1) * m_1 + (Relu(x + b_2) - Relu(x + b_1)) * m_2 + (Relu(-x-4)) * 20
    
    
    def potassium(x,b_1,b_2,m_1,m_2):
        return Relu(x + b_1) * m_1 + (Relu(x + b_2) - Relu(x + b_1)) * m_2
    
    # Assume x and u are 2D arrays.
    dx = np.zeros(x.shape)
    n = x.shape[0]

    # if para==None:
    #     b_1,b_2,m_1,m_2,n_1,n_2,x_f,x_s,c=np.array(np.ones(n)*1.0, np.ones(n)*3.0,np.ones(n)*20.0,np.ones(n)*1.0, np.ones(n)*70.0,np.ones(n)*4.0,np.ones(n)*5,-np.ones(n)*3,np.ones(n)*1)
    # else:
    assert len(para)==9
    b_1,b_2,m_1,m_2,n_1,n_2,x_f,x_s,c=para

    # Compute dx[:,1] and dx[:,2] with a loop.
    
    dx[:, 1] = -x[:, 1] + x[:, 0]
    dx[:, 2] = (-x[:, 2] + x[:, 0]) / 20.0

    # Compute leak, fast positive and slow negative components in a vectorized way.
    leak = 2 * (x[:, 0] + 1)
    fast_positive = sodium(x[:, 1],b_1,b_2,m_1,m_2)
    slow_negative = potassium(x[:, 2],b_1,b_2,n_1,n_2)
    
    
    dx[:, 0] = (-leak[:] - fast_positive[:] * (x[:, 0] - x_f) - slow_negative[:] * (x[:, 0] - x_s) + u[:, 0])/c
    
    return dx

@numba.njit
def ss_Relu_bursting(x, u):

    def Sat(x):
        return x * (x > 0) - (x-1)*(x-1>0)

    def sodium_act(x):
        # x can be a scalar or an array.
        return (Relu(x-1)-Relu(x-3)) * 30 + (Relu(x +1) - Relu(x-1)) * 10 + (Relu(-x-4)) * 5

    def sodium_inact(x):

        return (1-Sat(x-1))


    def potassium(x):

        return  Relu(x+1) * 10 

    def calcium(x):

        return Relu(-x-1)-Relu(-x-1.5)
    # Assume x and u are 2D arrays.
    dx = np.zeros(x.shape)
    n = x.shape[0]
    
    # Compute dx[:,1] and dx[:,2] with a loop.

    dx[:,1]=(-x[:,1]+x[:,0])/0.1
    dx[:,2]=(-x[:,2]+x[:,0])/5
    dx[:,3]=(-x[:,3]+x[:,0])/100

    leak=2*(x[:,0]+1)

    fast_positive=sodium_act(x[:,1])*sodium_inact(x[:,2])
    slow_negative=potassium(x[:,3])
    slow_positive=calcium(x[:,3])
    
    
    dx[:, 0] = -leak[:] - fast_positive[:] * (x[:, 0] - 7) - slow_negative[:] * (x[:, 0] + 3) - slow_positive*(x[:,0]-5) + u[:, 0] 
    
    return dx




@numba.njit
def syn_hh_numba(v, gain, res,alpha=1.5):
    # Compute synaptic effect on a single value.
    return gain / (1.0 + np.exp(-(v - res) * alpha))


@numba.njit(fastmath=True, parallel=False, cache=True)
def ring_ss_hh_vec_numba_impl(x, u, syn_strength, noise, inhibit_weight, current):
    B = x.shape[0]
    num = x.shape[1]
    dx = np.empty_like(x)
    # Preallocate u_eff once for reuse in each batch.
    u_eff = np.empty((num, 1))
    
    # Local constants for syn_hh parameters.
    const1 = -1.0
    const2 = -65.0

    for b in range(B):
        s_val = syn_strength[b]
        n_val = noise[b]
        i_val = inhibit_weight[b]
        c_val = current[b]
        
        # Cache the slice for this batch.
        xb = x[b]
        ub = u[b]
        
        t = np.empty(num)
        r = np.empty(num)
        s = 0.0
        # Precompute inhibition and synaptic connection values.
        for i in range(num):
            t[i] = syn_hh_numba(xb[i, 4], const1, const2)
            s += t[i]
            r[i] = syn_hh_numba(xb[i, 5], s_val, const2)
            
        # Compute a single noise term for this batch.
        noise_val = (np.random.random() - 0.5) * n_val
        
        # Build the effective input array using a separate handling for the first neuron.
        u_eff[0, 0] = c_val + ub[0, 0] + noise_val + (s - t[0]) * i_val + r[num - 1]
        for i in range(1, num):
            u_eff[i, 0] = c_val + ub[i, 0] + noise_val + (s - t[i]) * i_val + r[i - 1]
            
        # Pass only the HH state (first 4 columns) to ss_hh_numba.
        hh_state = xb[:, :4]
        dX_HH = ss_hh_numba(hh_state, u_eff)
        # Assign only the first 4 columns from dX_HH.
        dx[b, :num, 0:4] = dX_HH
        dx[b, :num, 4] = xb[:, 0] - xb[:, 4]
        dx[b, :num, 5] = (xb[:, 0] - xb[:, 5]) / 5.0
    return dx

@numba.njit(cache=True)
def ring_ss_hh_vec_numba(x, u, syn_strength, noise, inhibit_weight, current):
    """
    Wrapper that accepts scalars or 1D arrays for parameters.
    Scalars are broadcast to 1D arrays of length B.
    """
    B = x.shape[0]
    
    if isinstance(syn_strength, float):
        syn_strength = np.full(B, syn_strength)
    if isinstance(noise, float):
        noise = np.full(B, noise)
    if isinstance(inhibit_weight, float):
        inhibit_weight = np.full(B, inhibit_weight)
    if isinstance(current, float):
        current = np.full(B, current)
    
    return ring_ss_hh_vec_numba_impl(x, u, syn_strength, noise, inhibit_weight, current)


@numba.njit(fastmath=True, cache=True)
def ss_hh_numba(x, u):
    """
    Numba version of the Hodgkin–Huxley state derivatives.
    x: (N,4) array for HH state variables: [V, m, h, n]
    u: (N,1) external current per neuron.
    Returns dx: (N,4) time derivative of x.
    """
    N = x.shape[0]
    dx = np.empty_like(x)
    # Hodgkin–Huxley constants
    C   = 1.0
    gNa = 120.0
    gK  = 36.0
    gL  = 0.3
    ENa = 50.0
    EK  = -77.0
    EL  = -54.387

    for i in range(N):
        V = x[i, 0]
        m = x[i, 1]
        h = x[i, 2]
        n = x[i, 3]
        # Gating kinetics for m:
        a_m = 0.1 * (V + 40.0)
        exp_term_m = math.exp(-(V + 40.0) / 10.0)
        denom_m = 1.0 - exp_term_m
        if abs(denom_m) < 1e-6:
            alpha_m = 1.0
        else:
            alpha_m = a_m / denom_m
        beta_m = 4.0 * math.exp(-(V + 65.0) / 18.0)
        # For h:
        alpha_h = 0.07 * math.exp(-(V + 65.0) / 20.0)
        beta_h  = 1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0))
        # For n:
        a_n = 0.01 * (V + 55.0)
        exp_term_n = math.exp(-(V + 55.0) / 10.0)
        denom_n = 1.0 - exp_term_n
        if abs(denom_n) < 1e-6:
            alpha_n = 0.1
        else:
            alpha_n = a_n / denom_n
        beta_n = 0.125 * math.exp(-(V + 65.0) / 80.0)
        
        # Ionic currents:
        INa = gNa * m * m * m * h * (V - ENa)
        IK  = gK  * n * n * n * n * (V - EK)
        IL  = gL  * (V - EL)
        
        u_val = u[i, 0]
        dV = (u_val - INa - IK - IL) / C
        dm = alpha_m * (1.0 - m) - beta_m * m
        dh = alpha_h * (1.0 - h) - beta_h * h
        dn = alpha_n * (1.0 - n) - beta_n * n
        
        dx[i, 0] = dV
        dx[i, 1] = dm
        dx[i, 2] = dh
        dx[i, 3] = dn
    return dx

@numba.njit
def ring_ss_hh_vec_numba_center(x, u, syn_strength, noise, inhibit_weight, current):
    """
    Numba-compiled version of ring_ss_hh_vec.
    
    Parameters:
      x: Array of shape (B, num, 6) containing the state of B simulations.
         For each neuron:
           - Columns 0-3 hold the HH state,
           - Column 4 is used for inhibition,
           - Column 5 is used for the synaptic connection.
      u: Array of shape (B, num, 1) with external inputs.
      syn_strength, noise, inhibit_weight, current:
         Each can be a scalar or a 1D array; if scalar, it is used directly.
    
    Returns:
      dx: Array of shape (B, num, 6) with time derivatives.
    """
    B = x.shape[0]
    num = x.shape[1]
    dx = np.zeros_like(x)

    for b in range(B):
        # For each parameter, use scalar if provided; otherwise use per-b value.
        if isinstance(syn_strength, float):
            s_val = syn_strength
        else:
            s_val = syn_strength[b]
        if isinstance(noise, float):
            n_val = noise
        else:
            n_val = noise[b]
        if isinstance(inhibit_weight, float):
            i_val = inhibit_weight
        else:
            i_val = inhibit_weight[b]
        if isinstance(current, float):
            c_val = current
        else:
            c_val = current[b]
        
        # Compute inhibition term for batch b.
        # Compute synaptic connection for batch b.
        # Build the effective input u_eff for the HH dynamics.
        noise_val = (np.random.random() - 0.5) * n_val
        u_eff = np.empty((num, 1))
        s = sum(x[b, :, 4])/num
        for i in range(num):

            if i == 0:
                idx = num - 1
            else:
                idx = i - 1

            u_eff[i, 0] = c_val + u[b, i, 0] + noise_val + syn_hh_numba(s, -1.0, -65.0)+syn_hh_numba(x[b, i, 4], 1.0, -65.0) * i_val + syn_hh_numba(x[b, idx, 5], s_val, -65.0)
    
        # Prepare the HH state for batch b.
        x_HH = np.empty((num, 4))
        for i in range(num):
            for j in range(4):
                x_HH[i, j] = x[b, i, j]
        dX_HH = ss_hh_numba(x_HH, u_eff)
        for i in range(num):
            for j in range(4):
                dx[b, i, j] = dX_HH[i, j]
            dx[b, i, 4] = (-x[b, i, 4] + x[b, i, 0]) / 1.0
            dx[b, i, 5] = (-x[b, i, 5] + x[b, i, 0]) / 5.0
    return dx

def ss_muti(x,u,g_u,tau_u):
    dx=np.zeros(x.shape)
    dx[:,1]=(-x[:,1]+x[:,0])/1
    dx[:,2]=(-x[:,2]+x[:,0])/tau_u
    dx[:,0]=-0.1*x[:,0]+u[:,0]+(-g_u*u_ion(x[:,2])*x[:,0]-f2_ion(x[:,0])*x[:,0])*(1-Sigmoid(x[:,0],6,3))+(5-0.5*u_ion(x[:,1]-4)*x[:,0]-f_ion(x[:,0]-4)*x[:,0])*Sigmoid(x[:,0],6,3)
    
    return dx

# def ss_Relu(x,u):
#     dx=np.zeros(x.shape)
#     dx[:,1]=(-x[:,1]+x[:,0])/3
#     dx[:,2]=(-x[:,2]+x[:,0])/100

#     leak=0.1*x[:,0]

#     fast_postive=(Relu(x[:,0]-1) + Relu(x[:,0]-6))
#     ultra_slow_inactivation=(1-Relu(x[:,2]-1) -Relu(x[:,1]-6))
#     slow_negative=Relu(x[:,1]-6)
#     ultra_slow_negative=Relu(x[:,2]-1)

#     slow_postive=Relu(-x[:,2]+2)

#     dx[:,0]=-leak-fast_postive*ultra_slow_inactivation*(x[:,0]-15)-4*ultra_slow_negative*(x[:,0]+4)+u[:,0] - slow_negative*(x[:,0]-1) - slow_postive*(x[:,0]-2)

#     return dx

def ss_hh(x, u):
    # x is an array of shape (N,4): columns [V, m, h, n]
    # u is an array of shape (N,1) representing the external current (µA/cm²)
    
    dx = np.zeros(x.shape)
    
    # Extract state variables
    V = x[:, 0]
    m = x[:, 1]
    h = x[:, 2]
    n = x[:, 3]
    
    # HH constants (typical values)
    C   = 1.0       # membrane capacitance, µF/cm²
    gNa = 120.0     # maximum Na conductance, mS/cm²
    gK  = 36.0      # maximum K conductance, mS/cm²
    gL  = 0.3       # leak conductance, mS/cm²
    ENa = 50.0      # Na reversal potential, mV
    EK  = -77.0     # K reversal potential, mV
    EL  = -54.387   # leak reversal potential, mV

    # Vectorized gating rate functions:
    # For alpha_m: 0.1*(V+40)/(1 - exp(-(V+40)/10)), with limit=1.0 when V ~ -40 mV.
    a_m = 0.1 * (V + 40)
    exp_term_m = np.exp(-(V + 40) / 10)
    denom_m = 1 - exp_term_m
    alpha_m = np.where(np.abs(denom_m) < 1e-6, 1.0, a_m / denom_m)
    
    beta_m = 4.0 * np.exp(-(V + 65) / 18)
    
    alpha_h = 0.07 * np.exp(-(V + 65) / 20)
    beta_h  = 1.0 / (1 + np.exp(-(V + 35) / 10))
    
    # For alpha_n: 0.01*(V+55)/(1 - exp(-(V+55)/10)), with limit=0.1 when V ~ -55 mV.
    a_n = 0.01 * (V + 55)
    exp_term_n = np.exp(-(V + 55) / 10)
    denom_n = 1 - exp_term_n
    alpha_n = np.where(np.abs(denom_n) < 1e-6, 0.1, a_n / denom_n)
    
    beta_n = 0.125 * np.exp(-(V + 65) / 80)
    
    # Ionic currents (Ohm's law style)
    INa = gNa * (m ** 3) * h * (V - ENa)
    IK  = gK  * (n ** 4) * (V - EK)
    IL  = gL * (V - EL)
    
    # Compute derivative of membrane voltage
    dx[:, 0] = (u[:, 0] - INa - IK - IL) / C
    
    # Gating variable dynamics (first-order kinetics)
    dx[:, 1] = alpha_m * (1 - m) - beta_m * m  # dm/dt
    dx[:, 2] = alpha_h * (1 - h) - beta_h * h  # dh/dt
    dx[:, 3] = alpha_n * (1 - n) - beta_n * n  # dn/dt
    
    return dx

def ring_ss_hh_vec(num, x, u, syn_strength=0.1, noise=0, inhibit_weight=10, current=-1.0):
    """
    Vectorized version of ring_ss_hh that processes a batch of state arrays and
    allows vectorized parameters for syn_strength, noise, inhibit_weight, and current.
    
    Parameters:
      num: Number of neurons.
      x: State array of shape (B, num, 6), where for each neuron:
         - Columns 0-3 are for the HH model [V, m, h, n],
         - Column 4 is used for computing the inhibition term,
         - Column 5 is used for computing the synaptic connection.
      u: External input array of shape (B, num, 1)
      syn_strength: Gain for the synaptic connection. Can be a scalar or an array of shape (B,) (broadcasted to (B,1)).
      noise: Noise amplitude. Can be a scalar or an array of shape (B,) (broadcasted to (B,1,1)).
      inhibit_weight: Weight for the inhibition. Can be a scalar or an array of shape (B,) (broadcasted to (B,1)).
      current: Base current. Can be a scalar or an array of shape (B,) (broadcasted to (B,1,1)).
      
    Returns:
      dx: Time derivative of x with shape (B, num, 6)
    """
    B = x.shape[0]
    dx = np.zeros_like(x)
    
    # --- Vectorize the parameters so they broadcast correctly ---
    # syn_strength is used with an array of shape (B, num), so reshape to (B,1)
    syn_strength = np.array(syn_strength)
    if syn_strength.ndim == 0:
        syn_strength = np.full((B, 1), syn_strength)
    elif syn_strength.ndim == 1:
        if syn_strength.shape[0] != B:
            raise ValueError("syn_strength must have length equal to the batch size.")
        syn_strength = syn_strength.reshape(B, 1)
    
    # noise is used in an expression with shape (B, num, 1), so reshape to (B,1,1)
    noise = np.array(noise)
    if noise.ndim == 0:
        noise = np.full((B, 1, 1), noise)
    elif noise.ndim == 1:
        if noise.shape[0] != B:
            raise ValueError("noise must have length equal to the batch size.")
        noise = noise.reshape(B, 1, 1)
    
    # inhibit_weight is used with an array of shape (B, num), so reshape to (B,1)
    inhibit_weight = np.array(inhibit_weight)
    if inhibit_weight.ndim == 0:
        inhibit_weight = np.full((B, 1), inhibit_weight)
    elif inhibit_weight.ndim == 1:
        if inhibit_weight.shape[0] != B:
            raise ValueError("inhibit_weight must have length equal to the batch size.")
        inhibit_weight = inhibit_weight.reshape(B, 1)
    
    # current is used with shape (B, num, 1), so reshape to (B,1,1)
    current = np.array(current)
    if current.ndim == 0:
        current = np.full((B, 1, 1), current)
    elif current.ndim == 1:
        if current.shape[0] != B:
            raise ValueError("current must have length equal to the batch size.")
        current = current.reshape(B, 1, 1)
    
    # --- Connectivity: Inhibition and Synaptic Connection ---
    # Precompute the connectivity matrix (num x num)
    M = np.ones((num, num)) - np.eye(num)
    
    # Compute the inhibition term.
    # Use the helper syn_hh function on column 4 of x. (x[:,:,4] has shape (B, num))
    inhibit_raw = Syn_hh(x[:, :, 4], -1, -65)  # shape (B, num)
    inhibit = np.einsum('ij,bj->bi', M, inhibit_raw)  # shape (B, num)
    
    # Compute the synaptic connection using a rolled version of column 5.
    rolled = np.roll(x[:, :, 5], 1, axis=1)  # shape (B, num)
    # Here, syn_strength is now (B,1) and will be broadcast across neurons.
    syn_conn_raw = Syn_hh(rolled, syn_strength, -65)  # shape (B, num)
    
    # --- Build the effective input u_ ---
    # u is provided as (B, num, 1)
    # noise_term: generate noise with shape (B, num, 1) and multiply by noise (B,1,1)
    noise_term = (np.random.rand(B, num, 1) - 0.5) * noise
    # For inhibition and synaptic connection, add a singleton dimension to match (B, num, 1)
    inhibit_term = (inhibit * inhibit_weight)[:, :, np.newaxis]  # shape (B, num, 1)
    syn_conn_term = syn_conn_raw[:, :, np.newaxis]                # shape (B, num, 1)
    # current term is already (B,1,1) and will broadcast over neurons.
    u_ = current + u + noise_term + inhibit_term + syn_conn_term
    
    # --- Compute HH dynamics using ss_hh (which is already vectorized over neurons) ---
    # Flatten the batch dimension: X_flat will have shape (B*num, 4) and U_flat shape (B*num, 1)
    X_flat = x[:, :, 0:4].reshape(-1, 4)
    U_flat = u_.reshape(-1, 1)
    dX_flat = ss_hh(X_flat, U_flat)  # ss_hh returns derivatives for HH state variables, shape (B*num, 4)
    dx[:, :, 0:4] = dX_flat.reshape(B, num, 4)
    
    # --- Update the remaining state variables (columns 4 and 5) ---
    dx[:, :, 4] = (-x[:, :, 4] + x[:, :, 0]) / 1
    dx[:, :, 5] = (-x[:, :, 5] + x[:, :, 0]) / 5
    
    return dx



@numba.njit
def ss_Luka(x,u):
    dx=np.zeros(x.shape)
    # time-scale separation (slow and ultraslow)
    dx[:,1]=(-x[:,1]+x[:,0])/20.0      # tau_s = 50

    # passive leak
    leak=0.5*x[:,0]

    # localized conductance elements
    I_fast=-2*np.tanh(x[:,0]+1.5)  # fast negative conductance element
    I_slow= 2*np.tanh(x[:,1]+1.5)  # slow (+) and slow (–) combined
       # ultraslow restorative (+) element

    # membrane dynamics
    dx[:,0]=-leak-I_fast-I_slow+u[:,0]

    return dx

@numba.njit
def ss_Luka_slow(x,u,time=2.5,modulation=None):
    u_modulation=np.zeros(x.shape[0])
    if modulation is not None:
        if np.max(modulation)>1.5:
            print("modulation should be in the range of [0, 1.5]")
        else:
            u_modulation=modulation
    dx=np.zeros(x.shape)
    # time-scale separation (slow and ultraslow)
    dx[:,1]=(-x[:,1]+x[:,0])/3.0      # tau_s = 50
    dx[:,2]=(-x[:,2]+x[:,0])/20.0/time   # tau_us = 2500

    # passive leak
    leak=0.5*x[:,0]

    # localized conductance elements
    I_fast=-2.0*np.tanh(x[:,0]-3)                 # fast negative conductance element
    I_slow= 2.0*np.tanh(x[:,1]-3) -1.5*np.tanh(x[:,1]+1.5)  # slow (+) and slow (–) combined
    I_ultraslow=-(1.5-u_modulation)*np.tanh(x[:,2]+1.5)            # ultraslow restorative (+) element

    # membrane dynamics
    dx[:,0]=-leak-I_fast-I_slow+u[:,0]+I_ultraslow

    return dx

# Rebound bursting simulation for the requested ODE form

# ODE in the user's requested form
@numba.njit
def ss_Luka_bursting(x,u,modulation=None):
    dx=np.zeros(x.shape)
    u_modulation=np.zeros(x.shape[0])
    if modulation is not None:
        if np.max(modulation)>1.5:
            print("modulation should be in the range of [0, 1.5]")
        else:
            u_modulation=modulation

    # time-scale separation (slow and ultraslow)
    dx[:,1]=(-x[:,1]+x[:,0])/5.0      # tau_s = 50
    dx[:,2]=(-x[:,2]+x[:,0])/150.0    # tau_us = 2500

    # passive leak
    leak=0.5*x[:,0]

    # localized conductance elements
    I_fast=-2.0*np.tanh(x[:,0]-3)                 # fast negative conductance element
    I_slow= 2.0*np.tanh(x[:,1]-3) - (1.5-u_modulation*2)*np.tanh(x[:,1]+1.5)   # slow (+) and slow (–) combined
    I_ultraslow=-(1.5+u_modulation)*np.tanh(x[:,2]+1.5)            # ultraslow restorative (+) element

    # membrane dynamics
    dx[:,0]=-leak-I_fast-I_slow+u[:,0]+I_ultraslow

    return dx

@numba.njit
def ss_Luka_topology_multi_syn_multi_neuron(num, x, u, syn_topos,shifts,syn_taus,alphas,neuron_types,noise=0.0, current=-1.0,modulation=None,time=2.5,modulation_motor=None):
    # Allocate result arrays
    dx = np.zeros((num, 3+syn_topos.shape[0]))
    u_ = np.empty((num, 1))
    connections = np.zeros(num)

    assert x.shape[1] == 3+syn_topos.shape[0], "x must have same columns as 3+syn_topos.shape[0]"
    assert syn_topos.shape[1]==syn_topos.shape[2] == num, "syn_topos must be a square matrix of size num"
    assert syn_topos.shape[0] == shifts.shape[0] == syn_taus.shape[0] == alphas.shape[0], "All input arrays must have the same number of elements"
    assert neuron_types.shape[1] == num, "neuron_types must have same number of columns as num"
    assert neuron_types.shape[0] == 3, "neuron_types must have 3 rows"

    # Compute synaptic contributions using already numba-compiled functions
    for i in range(syn_topos.shape[0]):
        connections+=syn_topos[i,:,:] @ Syn_2((x[:, i+3]- shifts[i,:])*alphas[i,:])
        dx[:, 3+i] = (x[:, 0] - x[:, 3+i]) / syn_taus[i,:]
    
    # Compute random noise for each element in a vectorized manner
    rand_vals = np.random.random(num)
    u_[:, 0] = current + u + (rand_vals - 0.5) * noise + connections
    
    # Call the numba-compiled steady-state HH function and store its output

    if modulation is not None:
        u_modulation=modulation
    else:
        u_modulation=np.zeros(x.shape[0])

    if modulation_motor is not None:
        u_modulation_motor=modulation_motor
    else:
        u_modulation_motor=np.zeros(x.shape[0])

    dx[:, :3] = ss_Luka(x[:, :3], u_)*neuron_types[0,:].reshape(num,1)+ss_Luka_slow(x[:, :3], u_,time,u_modulation_motor)*neuron_types[1,:].reshape(num,1)+ss_Luka_bursting(x[:, :3], u_,u_modulation)*neuron_types[2,:].reshape(num,1)
    
    return dx


def ss_hr_deriv(x, u):
    """
    Vectorized differential equation function for the Hindmarsh–Rose bursting neuron model.
    
    State variables (per neuron):
      x[:,0] : Membrane potential (x)
      x[:,1] : Fast recovery variable (y)
      x[:,2] : Slow adaptation variable (z)
    
    External input:
      u[:,0] : External current (I_ext)
    
    The HR model equations are:
      dx/dt = y - a*x^3 + b*x^2 - z + I_ext
      dy/dt = c - d*x^2 - y
      dz/dt = r*( s*(x - x_R) - z)
      
    Parameter values (typical):
      a = 1.0, b = 3.0, c = 1.0, d = 5.0,
      r = 0.005, s = 4.0, x_R = -1.6.
    """
    # Create an array for the derivatives with the same shape as x.
    dx = np.zeros_like(x)
    
    # Parameters
    a = 1.0
    b = 3.0
    c = 1.0
    d = 5.0
    r = 0.005
    s = 4.0
    x_R = -1.6
    
    # Compute the derivatives in vectorized form:
    # Equation for x (membrane potential)
    dx[:, 0] = x[:, 1] - a * (x[:, 0] ** 3) + b * (x[:, 0] ** 2) - x[:, 2] + u[:, 0]
    # Equation for y (fast recovery variable)
    dx[:, 1] = c - d * (x[:, 0] ** 2) - x[:, 1]
    # Equation for z (slow adaptation variable)
    dx[:, 2] = r * (s * (x[:, 0] - x_R) - x[:, 2])
    
    return dx



def ring_ss(num,x,u,syn_strength=0.2,current=0.7,inhibit_weight=6,noise=0):
    u_=np.zeros((num,1))
    inhibit=(np.ones((num,num))-np.eye(num)) @ Syn(x[:,1],-1,1)
    syn_conn=Syn(np.roll(x[:,2],1),syn_strength,1)

    u_[:,0]=np.ones(num)*current +u + (np.random.rand(num)-0.5)*noise +inhibit*inhibit_weight+syn_conn

    dx=np.copy(ss_muti(x,u_,1,25))

    return dx

def ring_ss_Relu(num,x,u,syn_strength=0.2,current=0.7,inhibit_weight=6,noise=0,reverse_n=-2,reverse_p=-1):
    u_=np.zeros((num,1))
    inhibit=(np.ones((num,num))-np.eye(num)) @ Syn(x[:,0],-1,reverse_n)
    syn_conn=Syn(np.roll(x[:,2],1),syn_strength,reverse_p)

    u_[:,0]=np.ones(num)*current +u+ (np.random.rand(num)-0.5)*noise +inhibit*inhibit_weight+syn_conn

    dx=np.copy(ss_Relu(x,u_))

    return dx

def ring_ss_Luka(num,x,u,syn_strength=0.1,noise=0.1,inhibit_weight=3,current=0.1,reverse_n=-4,reverse_p=-4):
    u_=np.zeros((num,1))
    inhibit=(np.ones((num,num))-np.eye(num)) @ Syn2(x[:,0],-1,reverse_n)
    syn_conn=Syn2(np.roll(x[:,2],1),syn_strength,reverse_p)

    u_[:,0]=np.ones(num)*current +u+ (np.random.rand(num)-0.5)*noise +inhibit*inhibit_weight+syn_conn

    dx=np.copy(ss_Luka(x,u_))

    return dx

def ring_ss_hetero(num,x,u,g=1,tau=25,syn_strength=0.2,current=0.7,inhibit_weight=6,noise=0):
    u_=np.zeros((num,1))
    inhibit=(np.ones((num,num))-np.eye(num)) @ Syn(x[:,1],-1,1)
    syn_conn=Syn(np.roll(x[:,2],1),syn_strength,1)

    u_[:,0]=np.ones(num)*current +u+ (np.random.rand(num)-0.5)*noise +inhibit*inhibit_weight+syn_conn

    dx=np.copy(ss_muti(x,u_,g,tau))

    return dx


def ring_ss_hh(num,x,u,syn_strength=0.1,noise=0,inhibit_weight=10,current=-1.0,shift=-65):
    
    dx=np.zeros((num,6))
    
    u_=np.zeros((num,1))
    
    inhibit=(np.ones((num,num))-np.eye(num)) @ Syn_hh(x[:,4],-1,shift)
    
    syn_conn=Syn_hh(np.roll(x[:,5],1),syn_strength,shift)

    u_[:,0]=np.ones(num)*current +u + (np.random.rand(num)-0.5)*noise +inhibit*inhibit_weight+syn_conn

    dx[:,0:4]=np.copy(ss_hh(x[:,0:4],u_))
    
    dx[:,4]=(-x[:,4]+x[:,0])/1
    dx[:,5]=(-x[:,5]+x[:,0])/5

    return dx

def ring_ss_hh_center(num,x,u,syn_strength=0.1,noise=0,inhibit_weight=10,current=-1.0):
    
    dx=np.zeros((num,6))
    
    u_=np.zeros((num,1))
    
    inhibit= Syn_hh(sum(x[:,4])/num,-1,-65)+Syn_hh(x[:,4],1,-60)
    
    syn_conn=Syn_hh(np.roll(x[:,5],1),syn_strength,-65)

    u_[:,0]=np.ones(num)*current +u + (np.random.rand(num)-0.5)*noise +inhibit*inhibit_weight+syn_conn

    dx[:,0:4]=np.copy(ss_hh(x[:,0:4],u_))
    
    dx[:,4]=(-x[:,4]+x[:,0])/1
    dx[:,5]=(-x[:,5]+x[:,0])/5

    return dx

"""new version of implementation"""

@numba.njit
def ss_hh_topology(num, x, u, excitation_syn, inhibition_syn, noise=0.0, current=-1.0, i_shift=-65,e_shift=-65,fast_syn_tau=1.0,slow_syn_tau=5.0):
    # Allocate result arrays
    dx = np.zeros((num, 6))
    u_ = np.empty((num, 1))
    
    # Compute synaptic contributions using already numba-compiled functions
    inhibit = inhibition_syn @ syn_hh_numba(x[:, 4], -1, i_shift)
    excitation = excitation_syn @ syn_hh_numba(x[:, 5], 1, e_shift)
    
    # Compute random noise for each element in a vectorized manner
    rand_vals = np.random.random(num)
    temp = current + u + (rand_vals - 0.5) * noise + inhibit + excitation
    # Fill u_ as a column vector
    for i in range(num):
        u_[i, 0] = temp[i]
    
    # Call the numba-compiled steady-state HH function and store its output
    dx[:, :4] = ss_hh_numba(x[:, :4], u_)
    
    # Simplify the difference calculations
    dx[:, 4] = (x[:, 0] - x[:, 4]) / fast_syn_tau
    dx[:, 5] = (x[:, 0] - x[:, 5]) / slow_syn_tau
    
    return dx

@numba.njit
def ss_hh_topology_multi_syn(num, x, u, syn_topos,shifts,syn_taus,alphas,noise=0.0, current=-1.0):
    # Allocate result arrays
    dx = np.zeros((num, 4+syn_topos.shape[0]))
    u_ = np.empty((num, 1))
    connections = np.zeros(num)

    assert x.shape[1] == 7, "x must have 7 columns"
    assert syn_topos.shape[1]==syn_topos.shape[2] == num, "syn_topos must be a square matrix of size num"
    assert syn_topos.shape[0] == shifts.shape[0] == syn_taus.shape[0] == alphas.shape[0], "All input arrays must have the same number of elements"
    
    # Compute synaptic contributions using already numba-compiled functions
    for i in range(syn_topos.shape[0]):
        connections+=syn_topos[i,:,:] @ syn_hh_numba(x[:, i+4], 1, shifts[i],alphas[i])
        dx[:, 4+i] = (x[:, 0] - x[:, 4+i]) / syn_taus[i]
    
    # Compute random noise for each element in a vectorized manner
    rand_vals = np.random.random(num)
    u_[:, 0] = current + u + (rand_vals - 0.5) * noise + connections
    
    # Call the numba-compiled steady-state HH function and store its output
    dx[:, :4] = ss_hh_numba(x[:, :4], u_)
    
    return dx



@numba.njit
def ss_Relu_topology(num, x, u, excitation_syn, inhibition_syn, noise=0.0, current=-1.0, i_shift=-1,e_shift=-1,act_syn_tau=1.0,inh_syn_tau=1.0,a_alpha=10,i_alpha=10,custom_syn_tau=1.0):
    # Allocate result arrays
    dx = np.zeros((num, 7))
    u_ = np.empty((num, 1))

    assert x.shape[1] == 7, "x must have 7 columns"
    
    # Compute synaptic contributions using already numba-compiled functions
    inhibit = inhibition_syn @ Syn(x[:, 4], -1, i_shift,i_alpha)
    excitation = excitation_syn @ Syn(x[:, 5], 1, e_shift,a_alpha)
    
    # Compute random noise for each element in a vectorized manner
    rand_vals = np.random.random(num)
    temp = current + u + (rand_vals - 0.5) * noise + inhibit + excitation
    # Fill u_ as a column vector
    for i in range(num):
        u_[i, 0] = temp[i]
    
    # Call the numba-compiled steady-state HH function and store its output
    dx[:, :4] = ss_Relu(x[:, :4], u_)
    
    # Simplify the difference calculations
    dx[:, 4] = (x[:, 0] - x[:, 4]) / act_syn_tau
    dx[:, 5] = (x[:, 0] - x[:, 5]) / inh_syn_tau
    dx[:, 6] = (x[:, 0] - x[:, 6]) / custom_syn_tau
    
    return dx

@numba.njit
def ss_Relu_topology_multi_syn(num, x, u, syn_topos,shifts,syn_taus,alphas,noise=0.0, current=-1.0):
    # Allocate result arrays
    dx = np.zeros((num, 4+syn_topos.shape[0]))
    u_ = np.empty((num, 1))
    connections = np.zeros(num)

    assert x.shape[1] == 4+syn_topos.shape[0], "x must have same columns as 4+syn_topos.shape[0]"
    assert syn_topos.shape[1]==syn_topos.shape[2] == num, "syn_topos must be a square matrix of size num"
    assert syn_topos.shape[0] == shifts.shape[0] == syn_taus.shape[0] == alphas.shape[0], "All input arrays must have the same number of elements"
    
    # Compute synaptic contributions using already numba-compiled functions
    for i in range(syn_topos.shape[0]):
        connections+=syn_topos[i,:,:] @ Syn(x[:, i+4], 1, shifts[i],alphas[i])
        dx[:, 4+i] = (x[:, 0] - x[:, 4+i]) / syn_taus[i]
    
    # Compute random noise for each element in a vectorized manner
    rand_vals = np.random.random(num)
    u_[:, 0] = current + u + (rand_vals - 0.5) * noise + connections
    
    # Call the numba-compiled steady-state HH function and store its output
    dx[:, :4] = ss_Relu(x[:, :4], u_)
    
    return dx

@numba.njit
def ss_Relu_topology_multi_syn_multi_neuron(num, x, u, syn_topos,shifts,syn_taus,alphas,neuron_types,noise=0.0, current=-1.0):
    # Allocate result arrays
    dx = np.zeros((num, 4+syn_topos.shape[0]))
    u_ = np.empty((num, 1))
    connections = np.zeros(num)

    assert x.shape[1] == 4+syn_topos.shape[0], "x must have same columns as 4+syn_topos.shape[0]"
    assert syn_topos.shape[1]==syn_topos.shape[2] == num, "syn_topos must be a square matrix of size num"
    assert syn_topos.shape[0] == shifts.shape[0] == syn_taus.shape[0] == alphas.shape[0], "All input arrays must have the same number of elements"
    assert neuron_types.shape[1] == num, "neuron_types must have same number of columns as num"
    assert neuron_types.shape[0] == 3, "neuron_types must have 3 rows"

    # Compute synaptic contributions using already numba-compiled functions
    for i in range(syn_topos.shape[0]):
        connections+=syn_topos[i,:,:] @ Syn_2((x[:, i+4]- shifts[i,:])*alphas[i,:])
        dx[:, 4+i] = (x[:, 0] - x[:, 4+i]) / syn_taus[i,:]
    
    # Compute random noise for each element in a vectorized manner
    rand_vals = np.random.random(num)
    u_[:, 0] = current + u + (rand_vals - 0.5) * noise + connections
    
    # Call the numba-compiled steady-state HH function and store its output

    dx[:, :4] = ss_Relu(x[:, :4], u_)*neuron_types[0,:].reshape(num,1)+ss_Relu_slow_spike(x[:, :4], u_)*neuron_types[1,:].reshape(num,1)+ss_Relu_bursting(x[:, :4], u_)*neuron_types[2,:].reshape(num,1)
    
    return dx


@numba.njit
def ss_Relu_topology_multi_syn_multi_neuron_para(num, x, u, syn_topos,shifts,syn_taus,alphas,neuron_types,noise=0.0, current=-1.0,paras=None):
    # Allocate result arrays
    dx = np.zeros((num, 4+syn_topos.shape[0]))
    u_ = np.empty((num, 1))
    connections = np.zeros(num)
    n=x.shape[1]

    assert x.shape[1] == 4+syn_topos.shape[0], "x must have same columns as 4+syn_topos.shape[0]"
    assert syn_topos.shape[1]==syn_topos.shape[2] == num, "syn_topos must be a square matrix of size num"
    assert syn_topos.shape[0] == shifts.shape[0] == syn_taus.shape[0] == alphas.shape[0], "All input arrays must have the same number of elements"
    assert neuron_types.shape[1] == num, "neuron_types must have same number of columns as num"
    assert neuron_types.shape[0] == 3, "neuron_types must have 3 rows"

    # Compute synaptic contributions using already numba-compiled functions
    for i in range(syn_topos.shape[0]):
        connections+=syn_topos[i,:,:] @ Syn_2((x[:, i+4]- shifts[i,:])*alphas[i,:])
        dx[:, 4+i] = (x[:, 0] - x[:, 4+i]) / syn_taus[i,:]
    
    # Compute random noise for each element in a vectorized manner
    rand_vals = np.random.random(num)
    u_[:, 0] = current + u + (rand_vals - 0.5) * noise + connections
    
    # Call the numba-compiled steady-state HH function and store its output

    dx[:, :4] = ss_Relu_para(x[:, :4], u_,paras)*neuron_types[0,:].reshape(num,1)+ss_Relu_slow_spike(x[:, :4], u_)*neuron_types[1,:].reshape(num,1)+ss_Relu_bursting(x[:, :4], u_)*neuron_types[2,:].reshape(num,1)
    
    return dx

def ring_topology_gen(num,syn_strength=0.1,inhibit_weight=10): 

    inhibition_matrix=(np.ones((num,num))-np.eye(num))*inhibit_weight 

    excitation_matrix= np.roll(np.eye(num), shift=-1, axis=1)*syn_strength 

    return inhibition_matrix,excitation_matrix

def ring_topology_revert_gen(num,syn_strength=0.1,inhibit_weight=10): 

    inhibition_matrix=(np.ones((num,num))-np.eye(num))*inhibit_weight 

    excitation_matrix= np.roll(np.eye(num), shift=1, axis=1)*syn_strength 

    return inhibition_matrix,excitation_matrix



@numba.njit
def simulation_hh(num_neuron,Num_sample,dt,x0,excitation_matrix,inhibition_matrix,noise,current,e_shift,i_shift,u):
    outputs = np.empty((Num_sample + 1, num_neuron, 6))
    outputs[0] = x0[:]
    x=x0[:]
    for i in range(Num_sample): 

        dx=ss_hh_topology(num_neuron,x,u[i,:],excitation_matrix,inhibition_matrix,noise,current,e_shift=e_shift,i_shift=i_shift) 

        x=x+dx*dt 

        outputs[i+1]=x 
        
    return outputs

@numba.njit
def simulation_Relu(num_neuron,Num_sample,dt,x0,excitation_matrix,inhibition_matrix,noise,current,e_shift,i_shift,u): 
    outputs = np.empty((Num_sample + 1, num_neuron, 6))
    outputs[0] = x0[:]
    x=x0[:]
    for i in range(Num_sample): 

        dx=ss_Relu_topology(num_neuron,x,u[i,:],excitation_matrix,inhibition_matrix,noise,current,e_shift=e_shift,i_shift=i_shift) 

        x=x+dx*dt 

        outputs[i+1]=x 
        
    return outputs


# ---------------------------------------------------------------------------
# Notebook helpers (lifted from the original notebooks)
# ---------------------------------------------------------------------------
@numba.njit
def ss_Luka(x, u):
    dx = np.zeros(x.shape)
    dx[:, 1] = (-x[:, 1] + x[:, 0]) / 20.0
    leak = 0.5 * x[:, 0]
    i_fast = -2.0 * np.tanh(x[:, 0] + 1.5)
    i_slow = 2.0 * np.tanh(x[:, 1] + 1.5)
    dx[:, 0] = -leak - i_fast - i_slow + u[:, 0]
    return dx


@numba.njit
def ss_Luka_slow(x, u, time: float = 5.0, modulation: float = 0.0):
    dx = np.zeros(x.shape)
    dx[:, 1] = (-x[:, 1] + x[:, 0]) / 3.0
    dx[:, 2] = (-x[:, 2] + x[:, 0]) / 20.0 / time

    leak = 0.5 * x[:, 0]
    i_fast = -2.0 * np.tanh(x[:, 0] - 3)
    i_slow = 2.0 * np.tanh(x[:, 1] - 3) - 1.5 * np.tanh(x[:, 1] + 1.5)
    i_ultraslow = -(1.5 - modulation) * np.tanh(x[:, 2] + 1.5)
    dx[:, 0] = -leak - i_fast - i_slow + u[:, 0] + i_ultraslow
    return dx



@numba.njit
def simulate_rebound(sys, T: float = 500.0, dt: float = 0.05, base_current: float = -1.5,
                     pulse_current: float = -5.0, t_on: float = 150.0, t_off: float = 300.0):
    n_steps = int(T / dt)
    t = np.arange(n_steps) * dt
    u = np.ones((n_steps, 1)) * base_current
    u[(t >= t_on) & (t < t_off), 0] = pulse_current

    x = np.zeros((1, 3))
    X = np.zeros((n_steps, 1, 3))
    for k in range(n_steps):
        dx = sys(x, u[k:k + 1, :])
        x = x + dt * dx
        X[k, :, :] = x
    return t, X, u


@numba.njit
def simulate_rebound_modulate(T: float = 500.0, dt: float = 0.05, base_current: float = -1.5,
                              pulse_current: float = -5.0, t_on: float = 150.0, t_off: float = 300.0,
                              modulation: float = 0.0):
    n_steps = int(T / dt)
    t = np.arange(n_steps) * dt
    u = np.ones((n_steps, 1)) * base_current
    u[(t >= t_on) & (t < t_off), 0] = pulse_current

    x = np.zeros((1, 3))
    X = np.zeros((n_steps, 1, 3))
    for k in range(n_steps):
        dx = ss_Luka_slow(x, u[k:k + 1, :], modulation=modulation)
        x = x + dt * dx
        X[k, :, :] = x
    return t, X, u


def _prepare_modulation(modulation, n_neurons):
    if modulation is not None:
        return modulation
    return np.zeros(n_neurons)


def simulation_Luka(num_neuron, Num_sample, dt, x0, input, syn_topos, shifts, syn_taus, alphas,
                    neuron_types, noise=0.0, current=-1.0, modulation=None):
    outputs = np.empty((Num_sample + 1, num_neuron, 3 + syn_topos.shape[0]))
    outputs[0] = x0[:]
    x = x0[:]
    u_modulation = _prepare_modulation(modulation, x.shape[0])
    for i in range(Num_sample):
        if 3000 < i < 3500:
            u = np.zeros((num_neuron))
            u[0] = 3.0
        else:
            u = np.ones((num_neuron)) * 0.0
        dx = ss_Luka_topology_multi_syn_multi_neuron(
            num_neuron, x, input + u, syn_topos, shifts, syn_taus, alphas,
            neuron_types, noise=noise, current=current, modulation=u_modulation
        )
        x = x + dx * dt
        outputs[i + 1] = x
    return outputs


def simulation_Luka_fre(num_neuron, Num_sample, dt, x0, input, syn_topos, shifts, syn_taus, alphas,
                        neuron_types, noise=0.0, current=-1.0, modulation=None):
    outputs = np.empty((Num_sample + 1, num_neuron, 3 + syn_topos.shape[0]))
    inputs = np.empty((Num_sample + 1, num_neuron, 1))
    outputs[0] = x0[:]
    inputs[0, :, 0] = current + input
    x = x0[:]
    u_modulation = _prepare_modulation(modulation, x.shape[0])

    for i in range(Num_sample):
        if 50 / dt < i < 70 / dt:
            u = np.zeros((num_neuron))
            u[0] = 3.0
        elif 1200 / dt < i:
            u = np.ones((num_neuron)) * 1.3
        else:
            u = np.ones((num_neuron)) * 0.0

        dx = ss_Luka_topology_multi_syn_multi_neuron(
            num_neuron, x, input + u, syn_topos, shifts, syn_taus, alphas,
            neuron_types, noise=noise, current=current, modulation=u_modulation
        )
        x = x + dx * dt
        outputs[i + 1] = x
        inputs[i + 1, :, 0] = current + input + u
    return outputs, inputs


def simulation_Luka_phase(num_neuron, Num_sample, dt, x0, input, syn_topos, shifts, syn_taus, alphas,
                          neuron_types, noise=0.0, current=-1.0, modulation=None):
    outputs = np.empty((Num_sample + 1, num_neuron, 3 + syn_topos.shape[0]))
    inputs = np.empty((Num_sample + 1, num_neuron, 1))
    outputs[0] = x0[:]
    inputs[0, :, 0] = current + input
    x = x0[:]
    u_modulation = _prepare_modulation(modulation, x.shape[0])

    for i in range(Num_sample):
        if 50 / dt < i < 70 / dt:
            u = np.zeros((num_neuron))
            u[0] = 3.0
        elif 600 / dt < i < 620 / dt:
            u = np.zeros((num_neuron))
            u[1] = 2.0
        elif 1200 / dt < i < 1220 / dt:
            u = np.zeros((num_neuron))
            u[1] = 2.0
        else:
            u = np.ones((num_neuron)) * 0.0

        dx = ss_Luka_topology_multi_syn_multi_neuron(
            num_neuron, x, input + u, syn_topos, shifts, syn_taus, alphas,
            neuron_types, noise=noise, current=current, modulation=u_modulation
        )
        x = x + dx * dt
        outputs[i + 1] = x
        inputs[i + 1, :, 0] = current + input + u
    return outputs, inputs


def simulation_Luka_switch(num_neuron, Num_sample, dt, x0, input, syn_topos, shifts, syn_taus, alphas,
                           neuron_types, noise=0.0, current=-1.0, modulation=None):
    outputs = np.empty((Num_sample + 1, num_neuron, 3 + syn_topos.shape[0]))
    inputs = np.empty((Num_sample + 1, num_neuron, 1))
    outputs[0] = x0[:]
    inputs[0, :, 0] = current + input
    x = x0[:]
    u_modulation = _prepare_modulation(modulation, x.shape[0])

    for i in range(Num_sample):
        if 50 / dt < i < 70 / dt:
            u = np.zeros((num_neuron))
            u[0] = 3.0
        elif 1200 / dt < i < 1220 / dt:
            u = np.zeros((num_neuron))
            u[4] = 2.0
        else:
            u = np.ones((num_neuron)) * 0.0

        dx = ss_Luka_topology_multi_syn_multi_neuron(
            num_neuron, x, input + u, syn_topos, shifts, syn_taus, alphas,
            neuron_types, noise=noise, current=current, modulation=u_modulation
        )
        x = x + dx * dt
        outputs[i + 1] = x
        inputs[i + 1, :, 0] = current + input + u
    return outputs, inputs


def simulation_Luka_two_network(num_neuron, Num_sample, dt, x0, input, syn_topos, shifts, syn_taus, alphas,
                                neuron_types, noise=0.0, current=-1.0, modulation=None):
    outputs = np.empty((Num_sample + 1, num_neuron, 3 + syn_topos.shape[0]))
    inputs = np.empty((Num_sample + 1, num_neuron, 1))
    outputs[0] = x0[:]
    inputs[0, :, 0] = current + input
    x = x0[:]
    u_modulation = _prepare_modulation(modulation, x.shape[0])

    for i in range(Num_sample):
        if 50 / dt < i < 80 / dt:
            u = np.zeros((num_neuron))
            u[0] = 3.0
        else:
            u = np.ones((num_neuron)) * 0.0

        dx = ss_Luka_topology_multi_syn_multi_neuron(
            num_neuron, x, input + u, syn_topos, shifts, syn_taus, alphas,
            neuron_types, noise=noise, current=current, modulation=u_modulation
        )
        x = x + dx * dt
        outputs[i + 1] = x
        inputs[i + 1, :, 0] = current + input + u
    return outputs, inputs


def simulation_Luka_rebound_network(num_neuron, Num_sample, dt, x0, input, syn_topos, shifts, syn_taus, alphas,
                                    neuron_types, noise=0.0, current=-1.0, modulation=None):
    outputs = np.empty((Num_sample + 1, num_neuron, 3 + syn_topos.shape[0]))
    inputs = np.empty((Num_sample + 1, num_neuron, 1))
    outputs[0] = x0[:]
    inputs[0, :, 0] = current + input
    x = x0[:]
    u_modulation = _prepare_modulation(modulation, x.shape[0])

    for i in range(Num_sample):
        if 1000 / dt < i < 1200 / dt:
            u = np.zeros((num_neuron))
            u[0:3] += -3.0
        else:
            u = np.ones((num_neuron)) * 0.0

        dx = ss_Luka_topology_multi_syn_multi_neuron(
            num_neuron, x, input + u, syn_topos, shifts, syn_taus, alphas,
            neuron_types, noise=noise, current=current, modulation=u_modulation
        )
        x = x + dx * dt
        outputs[i + 1] = x
        inputs[i + 1, :, 0] = current + input + u
    return outputs, inputs
