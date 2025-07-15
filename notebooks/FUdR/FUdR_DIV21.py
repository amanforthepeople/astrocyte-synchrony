# ---
from brian2 import *
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt     # DELETE
#matplotlib.use('agg')  # DELETE
import numpy as np
import plot_utils as pu

# ---

#Simulation reproductibility
np.random.seed(500)
seed(500)  # <- This makes the entire simulation reproducible
start_scope()

# Define MEA representation with 32x32 electrodes
size = 3.75 * mmeter  # Physical size of the lattice/grid
electrode_spacing = 100 * umeter
grid_size = 32  # 32x32 grid
electrode_positions = [(i * size / grid_size - size / 2,
                        j * size / grid_size - size / 2)
                       for i in range(grid_size) for j in range(grid_size)]

# General parameters
duration = 15*second          # Total simulation time
N_e = 1000                   # Number of excitatory neurons
N_i = 250                    # Number of inhibitory neurons
N_a = 4                   # Number of astrocytes

# Neuron parameters
E_l = -60*mV                 # Leak reversal potential
g_l = 9.99*nS                # Leak conductance
E_e = 0*mV                   # Excitatory synaptic reversal potential
E_i = -80*mV                 # Inhibitory synaptic reversal potential
C_m = 198*pF                 # Membrane capacitance
tau_e = 5*ms                 # Excitatory synaptic time constant
tau_i = 10*ms                # Inhibitory synaptic time constant
tau_r = 5*ms                 # Refractory period
I_ex = 100*pA                 # Low baseline External current
V_th = -50*mV                # Firing threshold
V_r = E_l                    # Reset potential

# Synapse parameters
rho_c = 0.005                # Synaptic vesicle-to-extracellular space volume ratio
Y_T = 500.*mmolar            # Total vesicular neurotransmitter concentration
Omega_c_base = 40/second          # Neurotransmitter clearance rate
U_0__star = 0.6              # Resting synaptic release probability
Omega_f = 3.33/second        # Synaptic facilitation rate
Omega_d = 2.0/second         # Synaptic depression rate
w_e = 0.05*nS                # Excitatory synaptic conductance
w_i = 1.0*nS                 # Inhibitory synaptic conductance

# Presynaptic receptors
O_G = 1.5/umolar/second      # Agonist binding (activating) rate
Omega_G = 0.5/(60*second)    # Agonist release (deactivating) rate

# Astrocyte parameters
# Calcium fluxes
O_P = 0.9*umolar/second      # Maximal Ca^2+ uptake rate by SERCAs
K_P = 0.05*umolar            # Ca2+ affinity of SERCAs
C_T = 2*umolar               # Total cell free Ca^2+ content
rho_A = 0.18                 # ER-to-cytoplasm volume ratio
Omega_C = 6/second           # Maximal rate of Ca^2+ release by IP_3Rs
Omega_L = 0.1/second         # Maximal rate of Ca^2+ leak from the ER

# IP_3R kinectics
d_1 = 0.13*umolar            # IP_3 binding affinity
d_2 = 1.05*umolar            # Ca^2+ inactivation dissociation constant
O_2 = 0.2/umolar/second      # IP_3R binding rate for Ca^2+ inhibition
d_3 = 0.9434*umolar          # IP_3 dissociation constant
d_5 = 0.08*umolar            # Ca^2+ activation dissociation constant

# IP_3 production
# Agonist-dependent IP_3 production
O_beta = 0.5*umolar/second   # Maximal rate of IP_3 production by PLCbeta
O_N = 0.3/umolar/second      # Agonist binding rate
Omega_N = 0.5/second         # Maximal inactivation rate
K_KC = 0.5*umolar            # Ca^2+ affinity of PKC
zeta = 10                    # Maximal reduction of receptor affinity by PKC

# Endogenous IP3 production
O_delta = 1.2*umolar/second  # Maximal rate of IP_3 production by PLCdelta
kappa_delta = 1.5*umolar     # Inhibition constant of PLC_delta by IP_3
K_delta = 0.05*umolar         # Ca^2+ affinity of PLCdelta

# IP_3 degradation
Omega_5P = 0.05/second       # Maximal rate of IP_3 degradation by IP-5P
K_D = 0.7*umolar             # Ca^2+ affinity of IP3-3K
K_3K = 1.0*umolar            # IP_3 affinity of IP_3-3K
O_3K = 4.5*umolar/second     # Maximal rate of IP_3 degradation by IP_3-3K

# IP_3 diffusion
F = 0.1*umolar/second        # GJC IP_3 permeability
I_Theta = 0.1*umolar         # Threshold gradient for IP_3 diffusion
omega_I = 0.01*umolar        # Scaling factor of diffusion

# Gliotransmitter release and time course
C_Theta = 0.5*umolar         # Ca^2+ threshold for exocytosis
Omega_A = 0.6/second         # Gliotransmitter recycling rate
U_A = 0.9                    # Gliotransmitter release probability
G_T = 600*mmolar             # Total vesicular gliotransmitter concentration
rho_e = 3.0e-3               # Astrocytic vesicle-to-extracellular volume ratio
Omega_e = 30/second          # Gliotransmitter clearance rate
alpha = 0.2                  # Gliotransmission nature

# Neurons Eqn
dt = 0.1*ms
defaultclock.dt = dt
sigma_noise = 1.1*mV/sqrt(ms) #Higher amplitude for stronger random spiking  # noise explicitly in volt/sqrt(second)

neuron_equations = '''
dv/dt = (g_l*(E_l - v) + g_e*(E_e - v) + g_i*(E_i - v) + I_astro)/C_m
        + sigma_noise*xi : volt (unless refractory)
dg_e/dt = -g_e/tau_e : siemens
dg_i/dt = -g_i/tau_i : siemens
I_astro : amp
x : meter (constant)
y : meter (constant)
'''

neurons = NeuronGroup(N_e + N_i, model=neuron_equations,
                      threshold='v>V_th', reset='v=V_r',
                      refractory='tau_r', method='euler')

# Random initial membrane potential values and conductances
exc_neurons = neurons[:N_e]
inh_neurons = neurons[N_e:]

# Arrange excitatory neurons in a grid
N_rows = int(sqrt(N_e))
N_cols = N_e/N_rows
grid_dist = (size / N_cols)

# Use a normal distribution centered at (0,0) with some spread
spread = size / 4  # Adjust spread for more clustering
exc_neurons.x = 'clip(randn() * spread, -size/2, size/2)'
exc_neurons.y = 'clip(randn() * spread, -size/2, size/2)'
inh_neurons.x = 'clip(randn() * spread, -size/2, size/2)'
inh_neurons.y = 'clip(randn() * spread, -size/2, size/2)'

# Random initial membrane potential values and conductances
neurons.v = 'E_l + rand()*(V_th-E_l)'
neurons.g_e = 'rand()*w_e'
neurons.g_i = 'rand()*w_i'

# Synapses Eqn
synapses_equations = '''
# Neurotransmitter
dY_S/dt = -Omega_c * Y_S : mmolar (clock-driven)
# Fraction of activated presynaptic receptors
dGamma_S/dt = O_G * G_A * (1 - Gamma_S) - Omega_G * Gamma_S : 1 (clock-driven)
# Usage of releasable neurotransmitter per single action potential:
du_S/dt = -Omega_f * u_S                                    : 1 (event-driven)
# Fraction of synaptic neurotransmitter resources available for release:
dx_S/dt = Omega_d *(1 - x_S)                                : 1 (event-driven)
U_0                                                         : 1
# released synaptic neurotransmitter resources:
r_S                                                         : 1
# gliotransmitter concentration in the extracellular space:
G_A                                                         : mmolar
# which astrocyte covers this synapse ?
astrocyte_index : integer (constant)
'''

synapses_action = '''
U_0 = (1 - Gamma_S) * U_0__star + alpha * Gamma_S  # Explicit astrocyte-dependent modulation
u_S += U_0 * (1 - u_S)
r_S = u_S * x_S
x_S -= r_S
Y_S += rho_c * Y_T * r_S  # Neurotransmitter released into extracellular space
g_e_post += w_e * r_S * (1 + Gamma_S)  # Astrocytic gliotransmission enhances synaptic efficacy
'''

# Neuron Connections
exc_syn = Synapses(exc_neurons, neurons, model=synapses_equations,
               on_pre=synapses_action+'g_e_post += w_e*r_S',
               method='linear')

exc_syn.connect(True, p=0.05)
exc_syn.x_S = 1.0
inh_syn = Synapses(inh_neurons, neurons, model=synapses_equations,
               on_pre=synapses_action+'g_i_post += w_i*r_S',
               method='linear')
inh_syn.connect(True, p=0.15)
inh_syn.x_S = 1.0

# Connect excitatory synapses to an astrocyte depending on the position of the post-synaptic neuron
N_rows_a = int(sqrt(N_a))
N_cols_a = N_a/N_rows_a
grid_dist = size / (N_rows_a)
exc_syn.astrocyte_index = ('int(x_post/grid_dist) + '
                       'N_cols_a*int(y_post/grid_dist)')

# Astrocytes
# The astrocyte emits gliotransmitter when its Ca^2+ concentration crosses a threshold
astro_equations = '''
# Fraction of activated astrocyte receptors:
dGamma_A/dt = O_N * Y_S * (1 - clip(Gamma_A,0,1)) -
          Omega_N*(1 + zeta * C/(C + K_KC)) * clip(Gamma_A,0,1) : 1
# Intracellular IP_3
dI/dt = J_beta + J_delta - J_3K - J_5P + J_coupling              : mmolar
J_beta = O_beta * Gamma_A                                        : mmolar/second
J_delta = O_delta/(1 + I/kappa_delta) * C**2/(C**2 + K_delta**2) : mmolar/second
J_3K = O_3K * C**4/(C**4 + K_D**4) * I/(I + K_3K)                : mmolar/second
J_5P = Omega_5P*I                                                : mmolar/second
# Diffusion between astrocytes:
J_coupling                                                       : mmolar/second

# Ca^2+-induced Ca^2+ release:
dC/dt = J_r + J_l - J_p                                   : mmolar
dh/dt = (h_inf - h)/tau_h                                 : 1
J_r = (Omega_C * m_inf**3 * h**3) * (C_T - (1 + rho_A)*C) : mmolar/second
J_l = Omega_L * (C_T - (1 + rho_A)*C)                     : mmolar/second
J_p = O_P * C**2/(C**2 + K_P**2)                          : mmolar/second
m_inf = I/(I + d_1) * C/(C + d_5)                         : 1
h_inf = Q_2/(Q_2 + C)                                     : 1
tau_h = 1/(O_2 * (Q_2 + C))                               : second
Q_2 = d_2 * (I + d_1)/(I + d_3)                           : mmolar

# Fraction of gliotransmitter resources available for release:
dx_A/dt = Omega_A * (1 - x_A) : 1
# gliotransmitter concentration in the extracellular space:
dG_A/dt = -Omega_e*G_A        : mmolar
# Neurotransmitter concentration in the extracellular space:
Y_S                           : mmolar
C_Theta : mmolar (constant)
O_delta : mmolar/second (constant)
# The astrocyte position in space
x : meter (constant)
y : meter (constant)
'''
glio_release = '''
G_A += rho_e * G_T * U_A * x_A
x_A -= U_A *  x_A
'''

# Astrocyte Eqn
astrocytes = NeuronGroup(N_a, astro_equations,
                     threshold='C>C_Theta',
                     refractory='C>C_Theta',
                     reset=glio_release,
                     method='rk4',
                     dt=1e-2*second)

astrocytes.C_Theta = '0.3*umolar + 0.2*umolar*rand()'
astrocytes.O_delta = '1.0*umolar/second + 0.5*umolar/second*rand()'

spread = size / 4  # Controls clustering size
astrocytes.x = 'clip(randn() * spread, -size/2, size/2)'
astrocytes.y = 'clip(randn() * spread, -size/2, size/2)'

# Add random initialization
astrocytes.C = 0.01*umolar
astrocytes.h = 0.9
astrocytes.I = 0.01*umolar
astrocytes.x_A = 1.0

# Astrocyte connections to Neuron network
ecs_astro_to_syn = Synapses(astrocytes, exc_syn,
                        'G_A_post = G_A_pre : mmolar (summed)')
ecs_astro_to_syn.connect('i == astrocyte_index_post')# Increase astro-to-synapse connection probability
ecs_syn_to_astro = Synapses(exc_syn, astrocytes,
                        'Y_S_post = Y_S_pre/N_incoming : mmolar (summed)')
ecs_syn_to_astro.connect('astrocyte_index_pre == j')

# Diffusion between astrocytes
astro_to_astro_equations = '''
delta_I = I_post - I_pre : mmolar
C_avg = (C_pre + C_post) / 2 : mmolar
scaling_factor = 1 / (1 + exp(-10 * ((C_avg / umolar) - (C_Theta / umolar)))) : 1  # Sigmoid dependence on calcium
J_coupling_post = -(1 + tanh((abs(delta_I) - I_Theta) / omega_I)) * sign(delta_I) * F * scaling_factor / 2 : mmolar/second (summed)
'''

astro_to_neuron = Synapses(astrocytes, neurons, '''
                           I_astro_post = g_astro * G_A_pre : amp (summed)''')
astro_to_neuron.connect('sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) < 150*um')
g_astro = 250*pA/mmolar #stronger astrocytic influence

astro_to_astro = Synapses(astrocytes, astrocytes,
                      model=astro_to_astro_equations)
# Connect to all astrocytes less than 50um away mimicking GJCs
astro_to_astro.connect('i != j and '
                   'sqrt((x_pre-x_post)**2 +'
                   '     (y_pre-y_post)**2) < 50*um')

# Astrocyte loss progression
astrocyte_loss_rate = 15  # Number of astrocytes lost per second
astrocyte_calcium_decay = 0.8  # Fraction of calcium-related flux retained per second
gliotransmitter_decay = 0.8  # Fraction of gliotransmitter release retained per second

# Initialize astrocytes and neurotransmitter clearance
astrocyte_count = N_a
Omega_c = Omega_c_base

# Function to update astrocyte loss and neurotransmitter clearance
def update_astrocytes(t):
    global astrocyte_count, Omega_c
    astrocyte_loss = int(astrocyte_loss_rate * t)
    astrocyte_count = max(0, N_a - astrocyte_loss)
    Omega_c = Omega_c_base * (astrocyte_count / N_a)

# Update astrocyte properties over time
def update_astrocyte_properties(t):
    global Omega_C, O_P, Omega_L, Omega_A, rho_e, d_2, O_delta
    decay_factor = astrocyte_calcium_decay ** t
    gliotransmitter_factor = gliotransmitter_decay ** t

    # Reduce calcium fluxes
    Omega_C *= decay_factor
    O_P *= decay_factor
    Omega_L *= decay_factor
    d_2 *= decay_factor
    O_delta *= decay_factor

    # Reduce gliotransmitter release and recycling rates
    Omega_A *= gliotransmitter_factor
    rho_e *= gliotransmitter_factor

# Monitors
exc_mon = SpikeMonitor(exc_neurons)
inh_mon = SpikeMonitor(inh_neurons)
ast_mon = SpikeMonitor(astrocytes)

### We record some additional data from a single excitatory neuron
ni = 50
# Record conductances and membrane potential of neuron ni
state_mon = StateMonitor(exc_neurons, ['v', 'g_e', 'g_i'], record=ni)
synapse_mon = StateMonitor(exc_syn, ['u_S', 'x_S'],
                       record=exc_syn[ni, :], when='after_synapses')

astro_mon_G_A = StateMonitor(astrocytes, 'G_A', record=True)
astro_mon_coupling = StateMonitor(astrocytes, 'J_coupling', record=True) #Monitor astro-astro calcium diffusion
astro_mon_C = StateMonitor(astrocytes, 'C', record=True) #Monitor astrocyte intracellular calcium

# Simulation run with periodic updates for astrocyte loss and function decline
for t in range(int(duration / second)):
    update_astrocytes(t)
    update_astrocyte_properties(t)
    run(1*second)
    #run(duration, report='text')

# Analyze MEA activity, Convert neuron_positions to have units of meters
mea_activity = np.zeros((grid_size, grid_size))  # 22x22 MEA grid resolution
neuron_positions = np.array([neurons.x[:], neurons.y[:]]) * meter  # Explicitly add units, necessary for consistent units
astro_positions = np.array([astrocytes.x[:], astrocytes.y[:]]) * meter

for i, pos in enumerate(electrode_positions):
    x, y = pos  # Electrode position

    # Neuron contribution
    distance = np.sqrt((neuron_positions[0, :] - x)**2 +
                       (neuron_positions[1, :] - y)**2)
    weights = 1 / (distance + 1e-6 * meter)
    spike_count = np.sum(weights[:N_e] * exc_mon.count[:]) + \
                  np.sum(weights[N_e:] * inh_mon.count[:])

    # Astrocyte contribution
    distance_astro = np.sqrt((astro_positions[0, :] - x)**2 +
                             (astro_positions[1, :] - y)**2)
    weights_astro = 1 / (distance_astro + 1e-6 * meter)
    gliotransmitter_influence = np.sum(weights_astro * astro_mon_G_A.G_A[:, -1] * meter)

    # Normalize astrocyte influence to make it dimensionless
    normalized_gliotransmitter_influence = gliotransmitter_influence / umolar

    # Combine neuron and astrocyte contributions
    scaling_factor = 10.0  # Increase if needed
    spike_count *= (1 + scaling_factor * normalized_gliotransmitter_influence) #Scale neuron activity by astrocytic influence

    mea_activity[i // grid_size, i % grid_size] = spike_count

# Normalize MEA activity
mea_activity /= np.percentile(mea_activity, 95)  # Normalize to 95th percentile
mea_activity = np.clip(mea_activity, 0, 1)  # Cap values between 0 and 1

#Print Results
print(f"Number of spikes from excitatory neurons: {exc_mon.num_spikes}")
print(f"Number of spikes from inhibitory neurons: {inh_mon.num_spikes}")
print(f"Total excitatory synaptic connections: {len(exc_syn)}")
print(f"Total inhibitory synaptic connections: {len(inh_syn)}")
print(f"Connected astrocytes to synapses: {len(ecs_astro_to_syn)}")
print(f"Number of astro-to-astro connections: {len(astro_to_astro)}")
print(f"Remaining astrocytes: {astrocyte_count}")

# ---
import numpy as np
import matplotlib.pyplot as plt

# Extract spike times from the SpikeMonitor
spike_times = np.array(exc_mon.t / ms)  # Convert times to milliseconds
neuron_ids = np.array(exc_mon.i)  # Corresponding neuron indices

# Bin spikes into small time windows
bin_size = 5  # ms
max_time = int(duration / ms)
num_bins = max_time // bin_size
binned_spikes = np.zeros((N_e, num_bins))

for i, t in zip(neuron_ids, spike_times):
    bin_idx = int(t // bin_size)
    if bin_idx < num_bins:
        binned_spikes[i, bin_idx] += 1

# Compute population rate
population_rate = np.mean(binned_spikes, axis=0)

# Compute pairwise synchrony index
def compute_synchrony_index(spike_matrix):
    num_neurons, num_bins = spike_matrix.shape
    synchrony_matrix = np.zeros((num_neurons, num_neurons))

    for i in range(num_neurons):
        for j in range(i+1, num_neurons):
            spikes_i = spike_matrix[i, :]
            spikes_j = spike_matrix[j, :]

            # Compute the correlation coefficient between neuron pairs
            if np.sum(spikes_i) > 0 and np.sum(spikes_j) > 0:
                corr = np.corrcoef(spikes_i, spikes_j)[0, 1]
                synchrony_matrix[i, j] = corr
                synchrony_matrix[j, i] = corr

    # Return the average synchrony index
    return np.nanmean(synchrony_matrix)

synchrony_index = compute_synchrony_index(binned_spikes)
print(f"Synchrony Index: {synchrony_index:.4f}")

# Plot population activity
plt.figure(figsize=(10, 4))
plt.plot(np.arange(num_bins) * bin_size, population_rate, label="Population Rate")
plt.xlabel("Time (ms)")
plt.ylabel("Spike Rate")
plt.title("Population Activity Over Time")
plt.legend()
plt.show()

# ---
# Plot MEA activity
plt.figure(figsize=(8, 8))
plt.imshow(mea_activity, cmap='hot', interpolation='nearest', origin='lower')
plt.colorbar(label="Normalized MEA Activity")
plt.title("Normalized MEA Activity")
plt.xlabel("Electrode X Index")
plt.ylabel("Electrode Y Index")
plt.show()

# ---
#Neuron-Astrocyte Electrode Spacing
plt.scatter(exc_neurons.x[:] / meter, exc_neurons.y[:] / meter, s=1, label='Neurons')
plt.scatter(astrocytes.x[:] / meter, astrocytes.y[:] / meter, color = 'orange', s=2, label='Astrocytes')
plt.scatter([pos[0] for pos in electrode_positions],
            [pos[1] for pos in electrode_positions], c='red', label='Electrodes')
plt.legend()
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Neuron-Astrocyte Electrode Distribution")
plt.show()

# ---
plot(astro_mon_C.t / ms, astro_mon_C.C[0] / umolar)
xlabel('Time (ms)')
ylabel('Calcium Concentration (uM)')

# ---
# Plot G_A (Gliotransmitter Concentration) over time
plt.figure()
plt.title("Gliotransmitter Concentration (G_A) Over Time")
for i in range(4):
    plt.plot(astro_mon_G_A.t/ms, astro_mon_G_A.G_A[i]/mmolar, label=f'Astrocyte {i}')
plt.xlabel("Time (ms)")
plt.ylabel("G_A (mmolar)")
#plt.legend()
plt.show()

# ---
import matplotlib.pyplot as plt

# Time points from the monitor
time = astro_mon_C.t

# Indices of astrocytes to plot
astro_indices_to_plot = [0, 1, 2, 3]  # Change these indices to plot different astrocytes

plt.figure(figsize=(8, 4))

for astro_idx in astro_indices_to_plot:  # Loop through selected astrocytes
    calcium_signal = astro_mon_C.C[astro_idx]  # Access calcium signal
    coupling_signal = 2*(astro_mon_coupling.J_coupling[astro_idx])  # Access coupling signal

    # Plotting calcium signal
    plt.plot(time / second, calcium_signal / umolar, label=f'Calcium - Astrocyte {astro_idx}')
    # Plotting coupling signal
    plt.plot(time / second, coupling_signal / (umolar / second),
             label=f'Coupling - Astrocyte {astro_idx}', linestyle='--')

plt.title("Calcium and Coupling Dynamics for Selected Astrocytes")
plt.xlabel("Time (s)")
plt.ylabel("Concentration / Flux (μM or μM/s)")
#plt.legend()
plt.show()

# Print the data for the selected astrocytes
for astro_idx in astro_indices_to_plot:
    print(f"Astrocyte {astro_idx} coupling: {astro_mon_coupling.J_coupling[astro_idx]}")
    print(f"Astrocyte {astro_idx} calcium levels: {astro_mon_C.C[astro_idx]}")

# ---
# Plot Spiking Activity (Cell Index vs Time)
plt.figure(figsize=(8, 6))
plt.title("Spiking Activity: Cell Index vs Time")
#plt.plot(exc_mon.t[exc_mon.i <= N_e//4]/ms, exc_mon.i[exc_mon.i <= N_e//4], '|', color="blue", label="Excitatory")
#plt.plot(inh_mon.t[inh_mon.i <= N_i//4]/ms, inh_mon.i[inh_mon.i <= N_i//4]+N_e//4, '|', color="red", label="Inhibitory")
#plt.plot(ast_mon.t[ast_mon.i <= N_i//1]/ms, ast_mon.i[ast_mon.i <= N_i//1]+(N_e+N_i)//4, '|', color="green", label="Astrocytes")
plt.plot(exc_mon.t/ms, exc_mon.i, '|', label='Excitatory Neurons')
plt.plot(inh_mon.t/ms, inh_mon.i + N_e, '|', label='Inhibitory Neurons')
plt.plot(ast_mon.t/ms, ast_mon.i + N_e + N_i, '|', label='Astrocytes')
plt.ylim(0, 2500)  # Limit time to 0 to 1000 ms
plt.xlabel("Time (ms)")
plt.ylabel("Cell Index")
#plt.legend()
plt.show()

# ---
import matplotlib.pyplot as plt
import numpy as np

# Raster data
exc_spike_times = np.array(exc_mon.t / second)
exc_spike_indices = np.array(exc_mon.i)

# Astrocyte calcium dynamics
astro_calcium = np.array(astro_mon_C.C / umolar)
astro_time = np.array(astro_mon_C.t / second)

# Gliotransmission threshold (assumed dynamic, sampled per astro)
astro_thresholds = astrocytes.C_Theta[:]
astro_thresholds_vals = astro_thresholds / umolar  # Convert to same units

# Sample a few astrocytes
n_traces = 10
astro_indices = np.linspace(0, astro_calcium.shape[0] - 1, n_traces, dtype=int)
astro_traces = astro_calcium[astro_indices, :]
astro_event_times = []

# Extract Ca²⁺ event times (when Ca²⁺ > C_Theta)
for idx in astro_indices:
    ca_trace = astro_calcium[idx]
    threshold = astro_thresholds_vals[idx]
    above = ca_trace > threshold
    crossings = np.where(np.diff(above.astype(int)) == 1)[0]
    times = astro_time[crossings]
    astro_event_times.append(times)

# Plot
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                        gridspec_kw={'height_ratios': [1, 1]})
plt.subplots_adjust(hspace=0.3)

# --- Raster Plot ---
axs[0].scatter(exc_spike_times, exc_spike_indices, s=1, color='black')
axs[0].set_ylabel("Neuron Index")
axs[0].set_title("Raster Plot of Excitatory Neurons")

# --- Astrocyte Ca2+ Traces + Events ---
for i, (trace, times) in enumerate(zip(astro_traces, astro_event_times)):
    offset = i * 2.5
    axs[1].plot(astro_time, trace + offset, label=f'Astro {astro_indices[i]}')
    axs[1].scatter(times, np.ones_like(times)*offset, color='orange', s=20, label=None)

axs[1].set_ylabel("Astrocyte [Ca2+] (offset)")
axs[1].set_xlabel("Time (s)")
axs[1].set_title("Astrocyte Calcium Dynamics + Event Markers")
plt.tight_layout()
plt.show()

# ---
# Plot Firing Rate (binned)
bin_size = 1*ms  # Bin size in ms
#bins = np.arange(0, 1000, bin_size)
spk_count, bin_edges = np.histogram(np.r_[exc_mon.t/ms, inh_mon.t/ms],
                                    int((duration)/ms))
rate = double(spk_count)/(N_e + N_i)/bin_size/Hz  # Rate in Hz
plt.figure(figsize=(6, 4))
plt.step(bin_edges[:-1], rate, '-', color="black")
#plt.xlim(0, 1000)  # Limit time to 0 to 1000 ms
plt.title("Firing Rate")
plt.xlabel("Time (ms)")
plt.ylabel("Rate (Hz)")
plt.show()

# ---
#Dynamics of a Single Neurons
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot Membrane Potential
axes[0].axhline(V_th/mV, color='red', linestyle=':', label="Threshold (V_th)")
axes[0].plot(state_mon.t/ms, state_mon.v[0]/mV, color='black')
axes[0].vlines(exc_mon.t[exc_mon.i == ni]/ms, V_th/mV, 0, color='black')
#axes[0].set_xlim(0, 1000)  # Limit time to 0 to 1000 ms
axes[0].set_title("Membrane Potential")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Membrane Potential (mV)")

spk_index = np.in1d(synapse_mon.t, exc_mon.t[exc_mon.i == ni])

# Calculate r_S at spike times
x_S_spike = synapse_mon.x_S[0][spk_index]
u_S_spike = synapse_mon.u_S[0][spk_index]
r_S_spike = x_S_spike * u_S_spike  # Synaptic release

#Plot r_s
axes[1].vlines(synapse_mon.t[spk_index] / ms, 0, r_S_spike,
           color='C0', linewidth=2, label='Synaptic release (r_S)')
axes[1].set_ylim(0, 0.3)
#axes[1].set_xlim(0, 1000)
axes[1].set_title("Synaptic Release Variable Over Time")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel('Synaptic release (r_S)')


plt.tight_layout()
plt.show()

# ---
# Function to calculate firing rate for a subset of neurons
def calculate_firing_rate_subset(spike_monitor, neuron_ids, duration, bin_size=10*ms):
    """Calculate firing rate for a subset of neurons with time bins."""
    spike_times = np.array(spike_monitor.t / ms)  # spike times in ms
    spike_indices = np.array(spike_monitor.i)  # neuron indices for spikes
    total_neurons = len(neuron_ids)  # total number of neurons in the subset

    # Filter spikes for the subset of neurons
    subset_mask = np.isin(spike_indices, neuron_ids)
    subset_spike_times = spike_times[subset_mask]

    # Calculate firing rate
    time_bins = np.arange(0, duration / ms, bin_size / ms)
    spike_hist, _ = np.histogram(subset_spike_times, bins=time_bins)
    firing_rate = spike_hist / (bin_size / second * total_neurons)
    return time_bins[:-1], firing_rate

# Define a smaller subset of neurons
subset_neurons_exc = range(0, 100)  # First 100 excitatory neurons
subset_neurons_inh = range(0, 100)  # First 100 inhibitory neurons

# Plot raster and firing rate for the subset
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Raster Plot: Subset of Excitatory Neurons
exc_subset_mask = np.isin(exc_mon.i, subset_neurons_exc)
axes[0, 0].plot(exc_mon.t[exc_subset_mask] / ms, exc_mon.i[exc_subset_mask], '.', markersize=2)
axes[0, 0].set_title("Raster Plot: Excitatory Neurons (Subset)")
axes[0, 0].set_xlabel("Time (ms)")
axes[0, 0].set_ylabel("Neuron Index")

# Firing Rate: Subset of Excitatory Neurons
time_bins, firing_rate_exc_subset = calculate_firing_rate_subset(exc_mon, subset_neurons_exc, duration)
axes[0, 1].plot(time_bins, firing_rate_exc_subset)
axes[0, 1].set_title("Firing Rate: Excitatory Neurons (Subset)")
axes[0, 1].set_xlabel("Time (ms)")
axes[0, 1].set_ylabel("Rate (Hz)")

# Raster Plot: Subset of Inhibitory Neurons
inh_subset_mask = np.isin(inh_mon.i, subset_neurons_inh)
axes[1, 0].plot(inh_mon.t[inh_subset_mask] / ms, inh_mon.i[inh_subset_mask], '.', markersize=2, color='orange')
axes[1, 0].set_title("Raster Plot: Inhibitory Neurons (Subset)")
axes[1, 0].set_xlabel("Time (ms)")
axes[1, 0].set_ylabel("Neuron Index")

# Firing Rate: Subset of Inhibitory Neurons
time_bins, firing_rate_inh_subset = calculate_firing_rate_subset(inh_mon, subset_neurons_inh, duration)
axes[1, 1].plot(time_bins, firing_rate_inh_subset, color='orange')
axes[1, 1].set_title("Firing Rate: Inhibitory Neurons (Subset)")
axes[1, 1].set_xlabel("Time (ms)")
axes[1, 1].set_ylabel("Rate (Hz)")

plt.tight_layout()
plt.show()

# ---
plt.figure()
plt.plot(state_mon.t/ms, state_mon.g_e[0]/nS, label="Excitatory conductance")
plt.plot(state_mon.t/ms, state_mon.g_i[0]/nS, label="Inhibitory conductance")
plt.legend()
#plt.ylim (0,2)
plt.xlabel("Time (ms)")
plt.ylabel("Conductance (nS)")
plt.title("Synaptic Conductances")
plt.show()

# ---
# Total simulation time in seconds
simulation_time_s = float(duration / second)

# Spikes per second for excitatory neurons
spikes_per_min_exc = (exc_mon.num_spikes / simulation_time_s) * 0.60

# Spikes per second for inhibitory neurons
spikes_per_min_inh = (inh_mon.num_spikes / simulation_time_s) * 0.60

print(f"Excitatory spikes per second: {spikes_per_min_exc:.2f}")
print(f"Inhibitory spikes per second: {spikes_per_min_inh:.2f}")

# ---
#Total bursts
def detect_bursts(spike_times,
                  T_start=0.1,   # [s]: threshold to *initiate* a burst
                  T_in=0.1,      # [s]: threshold to *continue* a burst
                  T_end=0.1,     # [s]: threshold to *terminate* a burst
                  min_spikes=3):
    """
    Detect bursts in a single neuron's spike train using a three-threshold
    approach:
        - T_start: burst initiates if 2 consecutive ISIs < T_start.
        - T_in: subsequent spikes remain in the burst if next ISI < T_in.
        - T_end: the burst ends if the next ISI >= T_end.
        - min_spikes: minimum number of spikes required for a valid burst.

    Returns:
        A list of bursts, where each burst is itself a list of spike times
        that belong to that burst.
    """
    if len(spike_times) < 2:
        return []  # Not enough spikes to form a burst

    # Convert spike_times from a Brian2 Quantity to float (seconds), if needed
    spike_times = np.array(spike_times)  # typically in seconds

    # Compute inter-spike intervals
    ISI = np.diff(spike_times)

    bursts = []
    current_burst = []

    i = 0  # index over spike_times
    while i < len(spike_times):
        # If we don't have enough spikes left to check an ISI, just break
        if i >= len(spike_times) - 1:
            # If there's a current burst in progress, finalize it
            if len(current_burst) >= min_spikes:
                bursts.append(current_burst)
            break

        # Check if we can initiate or continue a burst
        # 1) If not currently in a burst, test T_start with this spike + next
        if not current_burst:
            # Initiate a burst if 2 consecutive intervals are < T_start
            # That means ISI[i] and ISI[i+1] both < T_start, but watch array bounds
            if i < len(spike_times) - 2:
                if (ISI[i] < T_start) and (ISI[i+1] < T_start):
                    # Start a new burst at spike_times[i]
                    current_burst = [spike_times[i], spike_times[i+1], spike_times[i+2]]
                    i += 2  # We have included i, i+1, i+2 in current_burst
                else:
                    i += 1
            else:
                # Not enough ISIs left to check
                i += 1
        else:
            # We are inside a burst:
            # Check the next ISI to decide if we continue or terminate
            if ISI[i] < T_in:
                # Still in the burst
                current_burst.append(spike_times[i+1])
                i += 1
            else:
                # The next ISI >= T_in... check T_end
                if ISI[i] >= T_end:
                    # End the burst
                    if len(current_burst) >= min_spikes:
                        bursts.append(current_burst)
                    current_burst = []
                    i += 1
                else:
                    # If the ISI is between T_in and T_end, you could decide a partial condition
                    # but let's keep it simple and continue the burst (or end it) depending on your definition.
                    # For demonstration, let's just keep the same logic as T_in:
                    current_burst.append(spike_times[i+1])
                    i += 1

    # If we exit the while loop and there's a burst in progress:
    if current_burst and (len(current_burst) >= min_spikes):
        bursts.append(current_burst)

    return bursts

# ---------------------------------------------
# Example usage AFTER the simulation finishes:
# ---------------------------------------------

# Total simulation time in seconds
simulation_time_s = float(duration/second)

# Create arrays to store # of bursts per neuron
exc_bursts_count = np.zeros(N_e)  # for each excitatory neuron
inh_bursts_count = np.zeros(N_i)  # for each inhibitory neuron

# We'll store bursts for each excitatory neuron in a dictionary for reference
exc_all_bursts = {}

# Detect bursts for each excitatory neuron
for neuron_idx in range(N_e):
    spike_times = exc_mon.spike_trains()[neuron_idx]  # in seconds
    bursts = detect_bursts(spike_times,
                           T_start=0.05,   # example parameters
                           T_in=0.05,
                           T_end=0.1,
                           min_spikes=3)
    exc_bursts_count[neuron_idx] = len(bursts)
    exc_all_bursts[neuron_idx] = bursts

# Same for inhibitory neurons
inh_all_bursts = {}
for neuron_idx in range(N_i):
    spike_times = inh_mon.spike_trains()[neuron_idx]
    bursts = detect_bursts(spike_times,
                           T_start=0.05,
                           T_in=0.05,
                           T_end=0.1,
                           min_spikes=3)
    inh_bursts_count[neuron_idx] = len(bursts)
    inh_all_bursts[neuron_idx] = bursts

# Now we have burst counts per neuron; compute bursts per minute:
exc_total_bursts = np.sum(exc_bursts_count)
exc_bursts_per_min = (exc_total_bursts / simulation_time_s) * 60.0
inh_total_bursts = np.sum(inh_bursts_count)
inh_bursts_per_min = (inh_total_bursts / simulation_time_s) * 60.0

print("---- Excitatory Neurons ----")
print(f"Total bursts detected across all excitatory neurons: {exc_total_bursts}")
print(f"Excitatory bursts per minute (population-wide): {exc_bursts_per_min:.2f}")

print("\n---- Inhibitory Neurons ----")
print(f"Total bursts detected across all inhibitory neurons: {inh_total_bursts}")
print(f"Inhibitory bursts per minute (population-wide): {inh_bursts_per_min:.2f}")

# If you want the average bursts-per-minute *per neuron*:
mean_exc_bursts_per_min_per_neuron = np.mean(exc_bursts_count) / simulation_time_s * 60.0
mean_inh_bursts_per_min_per_neuron = np.mean(inh_bursts_count) / simulation_time_s * 60.0

print("\n---- Per-Neuron Averages ----")
print(f"Average bursts per minute per excitatory neuron: {mean_exc_bursts_per_min_per_neuron:.2f}")
print(f"Average bursts per minute per inhibitory neuron: {mean_inh_bursts_per_min_per_neuron:.2f}")

# ---
import numpy as np
import matplotlib.pyplot as plt

# Convert spike times from the SpikeMonitor to milliseconds
spike_times = np.array(exc_mon.t / ms)  # spike times in ms
spike_indices = np.array(exc_mon.i)     # corresponding neuron indices

# Compute mean ISI for each neuron
unique_neurons = np.unique(spike_indices)
mean_isis_per_neuron = []

for neuron in unique_neurons:
    neuron_spike_times = spike_times[spike_indices == neuron]
    if len(neuron_spike_times) > 1:
        isis = np.diff(neuron_spike_times)
        mean_isis = np.mean(isis)
        mean_isis_per_neuron.append(mean_isis)

mean_isis_per_neuron = np.array(mean_isis_per_neuron)

# Print summary statistics
print("Mean ISIs per neuron (ms):", mean_isis_per_neuron)
print("Number of neurons with >1 spike:", len(mean_isis_per_neuron))
print("Overall Mean ISI (ms):", np.mean(mean_isis_per_neuron))
print("Standard Deviation of Mean ISIs (ms):", np.std(mean_isis_per_neuron))

# Plot the histogram of mean ISIs across neurons
plt.figure(figsize=(8, 5))
plt.hist(mean_isis_per_neuron, bins=50, edgecolor='black', alpha=0.7)
plt.ylim (0, 250)
plt.xlim (0, 7000)
plt.xlabel("Mean Interspike Interval per Neuron (ms)")
plt.ylabel("Number of Neurons")
plt.title("Distribution of Mean ISIs Across Neurons")
plt.show()

# ---
import numpy as np
import matplotlib.pyplot as plt

# Extract spike data
spike_times = np.array(exc_mon.t / ms)
spike_indices = np.array(exc_mon.i)

# Compute mean ISI and spike count per neuron
unique_neurons = np.unique(spike_indices)
mean_isis = []
spike_counts = []

for neuron in unique_neurons:
    times = spike_times[spike_indices == neuron]
    count = len(times)
    if count > 1:
        isis = np.diff(times)
        mean_isis.append(np.mean(isis))
        spike_counts.append(count)

mean_isis = np.array(mean_isis)
spike_counts = np.array(spike_counts)

# Print summary
print(f"Number of neurons with >1 spike: {len(mean_isis)}")
print(f"Mean of mean ISIs: {np.mean(mean_isis):.2f} ms")
print(f"Mean spike count: {np.mean(spike_counts):.1f}")
print("Standard Deviation of Mean ISIs (ms):", np.std(mean_isis_per_neuron))

# Plot: Mean ISI vs Spike Count
plt.figure(figsize=(8, 5))
plt.scatter(spike_counts, mean_isis, alpha=0.6, color='mediumseagreen', edgecolor='black')
plt.xlabel("Spike Count")
plt.ylabel("Mean ISI (ms)")
plt.title("Mean ISI vs. Spike Count per Neuron")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Function to smooth spike trains using Gaussian kernel smoothing
def smooth_spikes(spike_monitor, neuron_indices, bin_size=10*ms, duration=15*second):
    bins = int(duration / bin_size)
    signals = np.zeros((len(neuron_indices), bins))
    for idx, neuron_id in enumerate(neuron_indices):
        spike_times = spike_monitor.spike_trains()[neuron_id]
        indices = (spike_times / bin_size).astype(int)
        signals[idx, indices] = 1
    # Smooth the spike trains
    kernel = np.exp(-np.linspace(-3, 3, 30)**2)  # Broader kernel
    kernel /= kernel.sum()
    smoothed_signals = np.array([np.convolve(sig, kernel, mode='same') for sig in signals])
    return smoothed_signals

# Function to compute the PLV between two signals
def compute_plv(signal1, signal2):
    phase1 = np.angle(hilbert(signal1))
    phase2 = np.angle(hilbert(signal2))
    phase_diff = np.exp(1j * (phase1 - phase2))
    return np.abs(np.mean(phase_diff))

# Function to compute the global PLV for multiple signals
def compute_global_plv(signals):
    n = signals.shape[0]
    phase_diff_matrix = []
    for i in range(n):
        for j in range(i + 1, n):
            phase1 = np.angle(hilbert(signals[i]))
            phase2 = np.angle(hilbert(signals[j]))
            phase_diff = np.exp(1j * (phase1 - phase2))
            phase_diff_matrix.append(np.abs(np.mean(phase_diff)))
    return np.mean(phase_diff_matrix)

# Parameters for smoothing and analysis
bin_size = 4 * ms  # Larger bin size for better signal alignment
duration = 15 * second  # Total simulation time
start_time = 2 * second  # Focused time window start
end_time = 8 * second  # Focused time window end

# Select neurons for PLV computation
neuron_indices = range(300)  # Use the first 300 neurons for global PLV
smoothed_signals = smooth_spikes(exc_mon, neuron_indices, bin_size=bin_size, duration=duration)

# Select time window for analysis
start_idx = int(start_time / bin_size)
end_idx = int(end_time / bin_size)
smoothed_signals_window = smoothed_signals[:, start_idx:end_idx]

# Normalize signals before PLV computation
# Normalize signals before PLV computation (safely)
max_vals = np.max(smoothed_signals_window, axis=1, keepdims=True)
max_vals[max_vals == 0] = 1  # Avoid division by zero for inactive neurons
normalized_signals = smoothed_signals_window / max_vals

# Compute pairwise PLV for two neurons in the focused time window
pairwise_plv = compute_plv(normalized_signals[100], normalized_signals[200])

# Compute global PLV for the selected neurons in the focused time window
global_plv = compute_global_plv(normalized_signals)

# Plot the results
plt.figure(figsize=(10, 8))

# Plot the smoothed signals for Neuron 0 in the focused time window
plt.subplot(4, 1, 1)
plt.plot(normalized_signals[100], label="Neuron 100")
plt.title("Smoothed Signal - Neuron 100 (Focused Time Window)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
#plt.ylim(0.9, 1)
plt.legend()
plt.grid()

# Plot the smoothed signals for Neuron 1 in the focused time window
plt.subplot(4, 1, 2)
plt.plot(normalized_signals[200], label="Neuron 200", color='orange')
plt.title("Smoothed Signal - Neuron 200 (Focused Time Window)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
#plt.ylim(0.9, 1)
plt.legend()
plt.grid()

# Plot pairwise PLV
plt.subplot(4, 1, 3)
plt.bar(['Pairwise PLV'], [pairwise_plv], color='blue')
plt.title("Pairwise Phase-Locking Value (PLV)")
plt.ylabel("PLV")
plt.ylim(0, 1)
plt.grid()

# Plot global PLV
plt.subplot(4, 1, 4)
plt.bar(['Global PLV'], [global_plv], color='green')
plt.ylim (0, 0.25)
plt.title("Global Phase-Locking Value (PLV)")
plt.ylabel("PLV")
plt.ylim(0, 1)
plt.grid()

plt.tight_layout()
plt.show()

print(f"Updated Pairwise PLV (Neuron 0 vs Neuron 1, Focused Window): {pairwise_plv}")
print(f"Updated Global PLV (First 300 Neurons, Focused Window): {global_plv}")

# ---
# Select two specific neurons for cross-correlation
neuron1_id = 100  # Index of the first neuron
neuron2_id = 200  # Index of the second neuron

# Extract spike times
spike_times_neuron1 = exc_mon.spike_trains()[neuron1_id]
spike_times_neuron2 = exc_mon.spike_trains()[neuron2_id]

# Create spike train histograms
bin_width = 1*ms  # Set the resolution for the spike train
time_bins = np.arange(0, duration/ms, bin_width/ms)  # Time bins in ms

spike_train1, _ = np.histogram(spike_times_neuron1/ms, bins=time_bins)
spike_train2, _ = np.histogram(spike_times_neuron2/ms, bins=time_bins)

# Compute cross-correlation
cross_corr = np.correlate(spike_train1, spike_train2, mode='full')
lags = np.arange(-len(spike_train1) + 1, len(spike_train1)) * bin_width/ms

# Define a smaller lag range (e.g., -50 ms to 50 ms)
lag_limit = 100  # in ms
lag_mask = (lags >= -lag_limit) & (lags <= lag_limit)
lags_limited = lags[lag_mask]
cross_corr_limited = cross_corr[lag_mask]

# Plot cross-correlation over the smaller lag range
plt.figure(figsize=(8, 5))
plt.plot(lags_limited, cross_corr_limited)
plt.xlabel('Lag (ms)')
plt.ylabel('Cross-Correlation')
plt.title('Cross-Correlation between Neuron {} and Neuron {}'.format(neuron1_id, neuron2_id))
plt.grid()
plt.show()

# ---
from scipy.ndimage import label

# Parameters
burst_isi_threshold = 75 * ms  # max inter-spike interval within burst
network_burst_window = 75 * ms  # time window to consider multiple spikes as network burst
network_burst_participation = 0.15  # fraction of neurons spiking to qualify as network burst

# Get spike times
spike_times = exc_mon.t[:]
spike_indices = exc_mon.i[:]
N_exc = N_e

# Step 1: Total bursts per neuron
from collections import defaultdict

burst_counts = defaultdict(int)

for neuron_id in np.unique(spike_indices):
    neuron_spike_times = spike_times[spike_indices == neuron_id]
    if len(neuron_spike_times) < 2:
        continue
    isi = np.diff(neuron_spike_times)
    burst_labels, _ = label(isi < burst_isi_threshold)
    counts = np.bincount(burst_labels)[1:]  # Ignore label 0 (non-burst)
    burst_counts[neuron_id] = np.sum(counts >= 2)  # At least 3 spikes
for neuron_id in np.unique(spike_indices):
    neuron_spike_times = spike_times[spike_indices == neuron_id]
    if len(neuron_spike_times) < 2:
        continue
    isi = np.diff(neuron_spike_times)
    burst_labels, num_bursts = label(isi < burst_isi_threshold)
    total_bursts += np.sum(np.bincount(burst_labels)[1:] >= 2)  # Only count if ≥2 ISIs (≥3 spikes)

# Step 2: Network bursts
bin_size = 5 * ms
n_bins = int(duration / bin_size)

histogram, _ = np.histogram(spike_times / second, bins=n_bins, range=(0, duration/second))
threshold = N_exc * network_burst_participation


# Binary burst detection: 1 if above threshold, else 0
binary_burst = (histogram >= threshold).astype(int)
network_burst_labels, num_network_bursts = label(binary_burst)

print(f"\n----- BURST ANALYSIS -----")
print(f"Total individual bursts detected: {total_bursts}")
print(f"Total network bursts detected: {num_network_bursts}")

# ---
import matplotlib.pyplot as plt

# Convert to sorted list
sorted_neurons = sorted(burst_counts.keys())
burst_values = [burst_counts[n] for n in sorted_neurons]

plt.figure(figsize=(12, 4))
plt.bar(sorted_neurons, burst_values, color='skyblue')
plt.xlabel('Neuron index')
plt.ylabel('Number of bursts')
plt.title('Total bursts per excitatory neuron')
plt.tight_layout()
plt.show()

# ---
# Create time bins for histogram
bin_edges = np.linspace(0, duration/second, n_bins + 1)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

plt.figure(figsize=(12, 4))
plt.bar(bin_centers, histogram, width=(bin_edges[1] - bin_edges[0]), label='Spikes per bin')

# Overlay threshold line
plt.axhline(threshold, color='red', linestyle='--', label='Network burst threshold')

# Shade network bursts
for i in range(1, network_burst_labels.max()+1):
    start = bin_centers[network_burst_labels == i][0]
    end = bin_centers[network_burst_labels == i][-1]
    plt.axvspan(start, end, color='orange', alpha=0.3)

plt.xlabel('Time (s)')
plt.ylabel('Spike count per bin')
plt.title('Network burst detection (binned spikes)')
plt.legend()
plt.tight_layout()
plt.show()

# ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

# --- Parameters ---
burst_window_ms = 75
min_spikes_for_burst = 3
min_fraction_for_network_burst = 0.25
bin_size_ms = 75

# --- Extract spike data ---
spike_times = np.array(exc_mon.t / ms)
spike_indices = np.array(exc_mon.i)
duration_ms = float(duration / ms)
n_exc = N_e

# --- Compute mean ISI and spike count per neuron ---
unique_neurons = np.unique(spike_indices)
mean_isis = []
spike_counts = []

for neuron in unique_neurons:
    times = spike_times[spike_indices == neuron]
    count = len(times)
    if count > 1:
        isis = np.diff(times)
        mean_isis.append(np.mean(isis))
        spike_counts.append(count)

mean_isis = np.array(mean_isis)
spike_counts = np.array(spike_counts)

# --- Detect isolated bursts per neuron ---
burst_counts = {}
for neuron in unique_neurons:
    times = spike_times[spike_indices == neuron]
    count = 0
    if len(times) >= min_spikes_for_burst:
        for i in range(len(times) - min_spikes_for_burst + 1):
            window = times[i:i + min_spikes_for_burst]
            if window[-1] - window[0] <= burst_window_ms:
                count += 1
    burst_counts[neuron] = count

# --- Prepare data for network burst detection ---
n_bins = int(duration_ms / bin_size_ms)
bin_edges = np.linspace(0, duration_ms, n_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
histogram, _ = np.histogram(spike_times, bins=bin_edges)

active_neurons = len(np.unique(spike_indices))
network_threshold = int(np.ceil(min_fraction_for_network_burst * active_neurons))

burst_mask = histogram >= network_threshold
network_burst_labels, num_bursts = label(burst_mask)

# --- Create combined figure ---
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# --- Plot 1: Mean ISI vs Spike Count ---
axs[0].scatter(spike_counts, mean_isis, alpha=0.6, color='mediumseagreen', edgecolor='black')
axs[0].set_xlabel("Spike Count")
axs[0].set_ylabel("Mean ISI (ms)")
axs[0].set_title("Mean ISI vs. Spike Count per Neuron")
axs[0].grid(True)

# --- Plot 2: Isolated Bursts per Neuron ---
sorted_neurons = sorted(burst_counts.keys())
burst_values = [burst_counts[n] for n in sorted_neurons]
axs[1].bar(sorted_neurons, burst_values, color='skyblue')
axs[1].set_xlabel('Neuron Index')
axs[1].set_ylabel('Burst Count')
axs[1].set_title(f'Total Bursts per Neuron (≥{min_spikes_for_burst} spikes in {burst_window_ms}ms)')

# --- Plot 3: Network Bursts ---
axs[2].bar(bin_centers / 1000, histogram, width=(bin_edges[1] - bin_edges[0]) / 1000, label='Spikes per bin')
axs[2].axhline(network_threshold, color='red', linestyle='--', label='Network burst threshold')

for i in range(1, num_bursts + 1):
    burst_bins = bin_centers[network_burst_labels == i]
    start = burst_bins[0] / 1000
    end = burst_bins[-1] / 1000
    axs[2].axvspan(start, end, color='orange', alpha=0.3)

axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Spike Count per Bin')
axs[2].set_title(f'Network Bursts (≥{network_threshold} neurons spiking in {bin_size_ms}ms)')
axs[2].legend()

# --- Layout ---
plt.tight_layout()
plt.show()

# ---


