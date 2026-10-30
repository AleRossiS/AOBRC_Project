import torch
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import l4casadi as l4c 
import warnings

# To ignore some warnings from libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from neural_network import NeuralNetwork


# --- GLOBAL CONFIGURATION ---

# 1. System Dimensions
STATE_DIM = 2    # [q, dq]
CONTROL_DIM = 1  # [tau]
HIDDEN_SIZE = 64 # Neurons in hidden layers

# 2. Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
MODEL_PATH = os.path.join(SCRIPT_DIR, 'value_function_model.pt')
DATA_PATH = os.path.join(SCRIPT_DIR, 'pendulum_dataset.npz')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. Physics Parameters
PHYSICS_PARAMS = {
    'm': 1.0,   # Mass [kg]
    'l': 1.0,   # Length [m]
    'g': 9.81   # Gravity [m/s^2]
}
DT = 0.05       # Time step [s]

# 4. MPC & Cost Weights
# Weights for state [q, dq] and control [u]
WEIGHTS_X = [5.0, 0.1] 
WEIGHTS_U = [0.01]

# 5. Targets
TARGET_STATE = [np.pi, 0.0] # Upright position
TARGET_CONTROL = [0.0]


# --- LOAD SCALING PARAMETERS ---
def get_scaling_parameters(filepath):
    """
    Loads the original dataset to retrieve normalization factors.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at {filepath}")
    
    data = np.load(filepath)
    X = data['inputs']
    Y = data['targets']

    # axis=0 computes the max for each column of X
    X_MAX = np.max(np.abs(X), axis=0)
    Y_MAX = np.max(np.abs(Y))

    return X_MAX, Y_MAX


# --- CREATE TERMINAL COST FROM MODEL ---
def get_learned_terminal_cost(x_max_val, y_max_val):
    """
    Loads the trained neural network model and creates a CasADi function.
    """
    print("--- Converting PyTorch model to CasADi function ---")
    
    # Load the trained PyTorch model
    model = NeuralNetwork(input_size=STATE_DIM, hidden_size=HIDDEN_SIZE, output_size=1)
    model.to(DEVICE)
    
    print(f"Running on device: {DEVICE}")
    
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"CRITICAL ERROR loading weights: {e}")
        return None
    
    # Create CasADi function
    model.create_casadi_function(robot_name='simple_pendulum',
                                 NN_DIR=SCRIPT_DIR,
                                 input_size=STATE_DIM,
                                 load_weights=False)

    # NB: This raw function expects normalized inputs [-1, 1]
    casadi_funct_form_nn = model.nn_func
    
    # Define symbolic variable for physical inputs
    x_sym = ca.MX.sym('x_phys', STATE_DIM) 
    
    # To do element-wise division for normalization (next code line)
    scale_factor = ca.DM(x_max_val.reshape(-1, 1))
    
    # Normalize inputs
    x_norm = x_sym / scale_factor
    
    # Get normalized output from NN
    y_norm = casadi_funct_form_nn(x_norm)
    
    # Denormalize output
    y_phys = y_norm * y_max_val
    
    return ca.Function('V_learned', [x_sym], [y_phys])


# --- MPC CLASS --- 
class MPC_Controller:
    def __init__(self, horizon, dt=DT, use_terminal_cost=False, terminal_cost_func=None):
        """
        Initializes the MPC Optimization Problem.
        """
        self.N = horizon
        self.dt = dt
        self.use_terminal_cost = use_terminal_cost
        self.terminal_cost_func = terminal_cost_func
        self.opti = ca.Opti()
        
        # - Variables & Parameters -
        self.X = self.opti.variable(STATE_DIM, self.N + 1) 
        self.U = self.opti.variable(CONTROL_DIM, self.N)
        
        self.X0_param = self.opti.parameter(STATE_DIM)      # Initial State
        self.X_targ_param = self.opti.parameter(STATE_DIM)  # Target State
        self.U_targ_param = self.opti.parameter(CONTROL_DIM)  # Target Control
        
        # Weights (Parameters)
        self.Wx_vect = self.opti.parameter(STATE_DIM) 
        self.Wu_vect = self.opti.parameter(CONTROL_DIM)
        
        # Warm-starting storage
        self.prev_X_sol = None
        self.prev_U_sol = None
        
        # - Physics (Pendulum Dynamics) -
        m = PHYSICS_PARAMS['m']
        l = PHYSICS_PARAMS['l']
        g = PHYSICS_PARAMS['g']
        
        # Symbolic variables
        x_sym = ca.MX.sym('x', STATE_DIM)
        u_sym = ca.MX.sym('u', CONTROL_DIM)
        q, dq = x_sym[0], x_sym[1]
        
        # Dynamics equations 
        ddq = -(g/l) * ca.sin(q) + u_sym[0]/(m * l**2)
        Xdot = ca.vertcat(dq, ddq)

        # ODE system
        ode = {'x': x_sym, 'p': u_sym, 'ode': Xdot}

        # Integrator
        F = ca.integrator('F', 'rk', ode, 0, self.dt)

        # - Cost Function & Constraints -
        J_running = 0
        J_final = 0
        
        Wx = ca.diag(self.Wx_vect)
        Wu = ca.diag(self.Wu_vect)

        # Initial constraint
        self.opti.subject_to(self.X[:, 0] == self.X0_param)

        for k in range(self.N):
            x_k = self.X[:, k]
            u_k = self.U[:, k]

            # Cost accumulation
            err_x = x_k - self.X_targ_param
            err_u = u_k - self.U_targ_param
            J_running += err_x.T @ Wx @ err_x + err_u.T @ Wu @ err_u
            
            # Dynamics constraint 
            x_next = F(x0=x_k, p=u_k)['xf']
            self.opti.subject_to(self.X[:, k+1] == x_next)

        # Terminal Cost
        if use_terminal_cost and terminal_cost_func is not None:
            J_final += terminal_cost_func(self.X[:, self.N])
            
        self.opti.minimize(J_running + J_final)
        
        # --- Solver Options ---
        p_opts = {'expand': False, 'print_time': False} 
        s_opts = {
            'max_iter': 500,    
            'print_level': 0,
            'sb': 'yes',
            'tol': 1e-4,
            'hessian_approximation': 'limited-memory',
        }

        self.opti.solver('ipopt', p_opts, s_opts)
        
        
    def solve_one_step(self, x_current, target=None):
        # Use Global Target if none provided
        if target is None:
            target = TARGET_STATE

        # Set Parameter Values
        self.opti.set_value(self.X0_param, x_current)
        self.opti.set_value(self.X_targ_param, target)
        self.opti.set_value(self.U_targ_param, TARGET_CONTROL) 
        
        # Set Weights Values
        self.opti.set_value(self.Wx_vect, WEIGHTS_X)
        self.opti.set_value(self.Wu_vect, WEIGHTS_U)
        
        # Warm-Starting
        if self.prev_X_sol is not None:
            x_guess = np.hstack((self.prev_X_sol[:, 1:], self.prev_X_sol[:, -1:]))
            u_guess = np.hstack((self.prev_U_sol[:, 1:], self.prev_U_sol[:, -1:]))
            self.opti.set_initial(self.X, x_guess)
            self.opti.set_initial(self.U, u_guess)
        
        try:
            sol = self.opti.solve()
            
            x_val = sol.value(self.X)
            u_val = sol.value(self.U)
            
            # To solve an error that occurred
            if u_val.ndim == 1: u_val = u_val.reshape(1, -1)
            if x_val.ndim == 1: x_val = x_val.reshape(2, -1)

            self.prev_X_sol = x_val
            self.prev_U_sol = u_val
            
            return u_val[0, 0] # First control action

        except Exception as e:
            print(f"SOLVER FAIL")
            #print(f"SOLVER FAIL: {e}") # Decomment to see full error 

            #If solver fails use the last computed solution
            # Takes the last values computed by the solver before failing
            x_debug = self.opti.debug.value(self.X)
            u_debug = self.opti.debug.value(self.U)
            
            # To solve an error that occurred
            if u_debug.ndim == 1: u_debug = u_debug.reshape(1, -1)
            if x_debug.ndim == 1: x_debug = x_debug.reshape(2, -1)
            
            # Store for warm start
            self.prev_X_sol = x_debug
            self.prev_U_sol = u_debug
            
            # Return the calculated control action (even if sub-optimal)
            return float(u_debug[0, 0])


# --- MPC SIMULATION ---
def run_full_MPC(controller, x0, steps=80):
    """
    Executes the full MPC simulation using Global Physics.
    
    Args:
        controller: MPC_Controller instance
        x0: np.ndarray, initial state
        steps: int, number of simulation steps
    Returns:
        traj_x: np.ndarray of shape (steps+1, STATE_DIM)
        traj_u: np.ndarray of shape (steps, CONTROL_DIM) 
    """
    traj_x = [x0]
    traj_u = [] 
    curr_x = x0
    
    # Physics for Simulation
    dt = DT
    m = PHYSICS_PARAMS['m']
    l = PHYSICS_PARAMS['l']
    g = PHYSICS_PARAMS['g']
    
    for _ in range(steps):
        u = controller.solve_one_step(curr_x) # Get first control action
        traj_u.append(u)
        
        # Apply dynamics (Simple Euler)
        ddq = -(g/l) * np.sin(curr_x[0]) + u/(m * l**2)
        curr_x = curr_x + np.array([curr_x[1], ddq]) * dt
        
        traj_x.append(curr_x)
        
    return np.array(traj_x), np.array(traj_u)


# --- ACTUAL COST FOR COMPARISON ---
def calculate_realized_cost(traj_x, traj_u):
    """
    Calculates the total cost accumulated during the real simulation.
    Uses GLOBAL weights to ensure fair comparison.
    
    Args:
        traj_x: np.ndarray of shape (T, STATE_DIM)
        traj_u: np.ndarray of shape (T, CONTROL_DIM)
    Returns:
        total_cost: float
    """
    
    x_target = np.array(TARGET_STATE)
    u_target = np.array(TARGET_CONTROL)
    
    # Use Global Weights
    Wx = np.diag(WEIGHTS_X) 
    Wu = np.diag(WEIGHTS_U)     
    
    total_cost = 0.0
    n_steps = len(traj_u)
    
    for k in range(n_steps):
        x_k = traj_x[k] 
        u_k = traj_u[k]
        
        err_x = x_k - x_target
        err_u = u_k - u_target
        
        step_cost = (err_x.T @ Wx @ err_x) + (err_u.T @ Wu @ err_u)
        total_cost += step_cost
        
    return float(total_cost)


# --- PENDULUM ANIMATION FUNCTION ---
def animate_three_pendulums(traj_list, titles, dt=DT):
    """
    Creates an animation with 3 side-by-side panels.
    Style: Blue Rod and Red Trace.
    """
    L = PHYSICS_PARAMS['l']
    
    # Setup Figure: 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Swing-Up Comparison: Parallel Trajectories", fontsize=16)

    lines = []
    traces = []
    time_texts = []
    
    # Configure axes
    for i, ax in enumerate(axes):
        ax.set_xlim(-1.5 * L, 1.5 * L)
        ax.set_ylim(-1.5 * L, 1.5 * L)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title(titles[i], fontsize=12, fontweight='bold')
        
        # Draw Target (vertical dashed line)
        ax.plot([0, 0], [0, L], 'k--', alpha=0.3)
        # Draw Pivot
        ax.plot(0, 0, 'ks', markersize=5)
        
        # --- STYLE ---
        line, = ax.plot([], [], 'o-', lw=3, color='blue', markersize=8, markeredgecolor='k')
        # Trace: Red, thin
        trace, = ax.plot([], [], '-', lw=1, color='red', alpha=0.5)
        # Time text
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        
        lines.append(line)
        traces.append(trace)
        time_texts.append(time_text)

    # Init function
    def init():
        for line, trace, txt in zip(lines, traces, time_texts):
            line.set_data([], [])
            trace.set_data([], [])
            txt.set_text('')
        return lines + traces + time_texts

    # Update function
    def update(frame):
        artists = []
        
        for i in range(3):
            traj = traj_list[i]
            
            # Handle trajectory length (stop at last frame if finished)
            idx = min(frame, len(traj) - 1)
            
            # Polar -> Cartesian
            q = traj[idx, 0]
            x = L * np.sin(q)
            y = -L * np.cos(q)
            
            # 1. Update Rod (from 0,0 to x,y)
            lines[i].set_data([0, x], [0, y])
            
            # 2. Update Trace (history of last 40 steps)
            trace_len = 40
            start_trace = max(0, idx - trace_len)
            q_hist = traj[start_trace : idx+1, 0]
            x_hist = L * np.sin(q_hist)
            y_hist = -L * np.cos(q_hist)
            traces[i].set_data(x_hist, y_hist)
            
            # 3. Update Text
            time_texts[i].set_text(f"t = {idx * dt:.2f}s")
            
            artists.extend([lines[i], traces[i], time_texts[i]])
            
        return artists

    # Calculate max duration
    max_frames = max([len(t) for t in traj_list])
    
    # Create Animation
    ani = animation.FuncAnimation(fig, update, frames=max_frames, 
                                  init_func=init, blit=True, interval=40)
    
    plt.tight_layout()
    plt.show()
    
    
    

# --- MAIN ---
def main():
    # Load Data Scaling Parameters
    try:
        X_MAX, Y_MAX = get_scaling_parameters(DATA_PATH)
    except FileNotFoundError as e:
        print(f"Errore: {e}") 
        return      

    # Terminal Cost Function for Neural Network
    terminal_cost_from_nn = get_learned_terminal_cost(X_MAX, Y_MAX)
    
    if terminal_cost_from_nn is None:
        return

    print("\n--- Initializing Controllers ---")
    
    M = 5   # Short horizon 
    N = 100 # Long horizon 
    
    # MPC 1: Short Horizon M, NO Neural Network
    print(f"[Case 1] Short Horizon (M={M}) WITHOUT Terminal Cost")
    mpc_M = MPC_Controller(horizon=M, use_terminal_cost=False)
    
    # MPC 2: Short Horizon M + Neural Network
    print(f"[Case 2] Short Horizon (M={M}) WITH Neural Terminal Cost")
    mpc_M_NN = MPC_Controller(horizon=M, use_terminal_cost=True, terminal_cost_func=terminal_cost_from_nn)
    
    # MPC 3: Long Horizon N+M, NO Neural Network (Benchmark)
    print(f"[Case 3] Long Horizon (N + M={M+N}) Benchmark")
    mpc_N_plus_M = MPC_Controller(horizon=N + M, use_terminal_cost=False)
    
    
    # --- Run Full Simulations for one initial state (To show an example) ---
    #x_init = np.array([0.1, 0.3]) # Can be changed to any other initial state
    x_init = np.array([5.78, -2.83])
    print("\n--- Starting Simulations & Calculating Costs for 1 Example Initial State ---")
    print(f"\nExample Initial State: {x_init}\n")
    
    # --- SIM MPC1 ---
    print("Running Case 1...")
    traj_M_x, traj_M_u = run_full_MPC(mpc_M, x_init)
    cost_M_ex = calculate_realized_cost(traj_M_x, traj_M_u)

    # --- SIM MPC2 ---
    print("Running Case 2...")
    traj_M_NN_x, traj_M_NN_u = run_full_MPC(mpc_M_NN, x_init)
    cost_M_NN_ex = calculate_realized_cost(traj_M_NN_x, traj_M_NN_u)

    # --- SIM MPC3 ---
    print("Running Case 3...")
    traj_N_plus_M_x, traj_N_plus_M_u = run_full_MPC(mpc_N_plus_M, x_init)
    cost_N_plus_M_ex = calculate_realized_cost(traj_N_plus_M_x, traj_N_plus_M_u)

    # --- Print Comparison Table ---
    print("--- Comparison Table ---")
    print("\n" + "="*60)
    print(f"{'SCENARIO':<35} | {'REALIZED COST':<15}")
    print("-" * 60)
    print(f"{'Case 1: Short Horizon (No Cost)':<35} | {cost_M_ex:10.2f}")
    print(f"{'Case 2: Short Horizon + NN Cost':<35} | {cost_M_NN_ex:10.2f}")
    print(f"{'Case 3: Long Horizon (Benchmark)':<35} | {cost_N_plus_M_ex:10.2f}")
    print("="*60 + "\n")

    # Plot Results of the Example
    print("--- Plotting Results of the example ---")
    plt.figure(figsize=(10, 6))
    
    plt.plot(traj_M_x[:, 0], 'r--', label=f'M no Cost (J={cost_M_ex:.1f})')
    plt.plot(traj_N_plus_M_x[:, 0], 'g-', alpha=0.5, linewidth=3, label=f'Benchmark (J={cost_N_plus_M_ex:.1f})')
    plt.plot(traj_M_NN_x[:, 0], 'b.-', label=f'M + NN Cost (J={cost_M_NN_ex:.1f})')
    
    plt.axhline(TARGET_STATE[0], color='k', linestyle=':', label='Target')
    
    plt.title("Comparison: MPCs Trajectories for the example Initial State")
    plt.xlabel("Time Steps")
    plt.ylabel("Angle q [rad]")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # --- Pendulums Animation ---
    print("\n--- Showing Animation ---")
    
    trajectories = [traj_M_x, traj_M_NN_x, traj_N_plus_M_x]
    
    titles = [
        f"1. M No Cost\nJ = {cost_M_ex:.1f}",
        f"2. M + NN Cost \nJ = {cost_M_NN_ex:.1f}",
        f"3. N +M (Benchmark) \nJ = {cost_N_plus_M_ex:.1f}"
    ]
    
    animate_three_pendulums(trajectories, titles)
    
    
    # --- Simulations over multiple random initial states ---
    print("\n--- Running Simulations over Multiple Random Initial States ---")
    N_sim = 30
    
    # Initialize costs storage
    # Nota: N_sim è sufficiente, l'esempio iniziale non serve salvarlo qui
    costs_M = np.zeros(N_sim + 1)
    costs_M_NN = np.zeros(N_sim + 1)
    costs_N_plus_M = np.zeros(N_sim + 1) # +1 to include also the example run
    
    # Store the example run costs
    costs_M[N_sim] = cost_M_ex
    costs_M_NN[N_sim] = cost_M_NN_ex
    costs_N_plus_M[N_sim] = cost_N_plus_M_ex

    for sim_idx in range(N_sim):

        # Set random initial state
        q_rand = np.random.uniform(0, 2*np.pi) #rad
        dq_rand = np.random.uniform(-6.0, 6.0) #rad/s
        x_init = np.array([q_rand, dq_rand])
        
        print(f"\nRun {sim_idx+1}/{N_sim} | Init: [{x_init[0]:.2f} rad, {x_init[1]:.2f} rad/s]")
    
        # Reset warm start for all controllers between simulations
        mpc_M.prev_X_sol,        mpc_M.prev_U_sol        = None, None
        mpc_M_NN.prev_X_sol,     mpc_M_NN.prev_U_sol     = None, None
        mpc_N_plus_M.prev_X_sol, mpc_N_plus_M.prev_U_sol = None, None
        
        # Case 1
        print("Running Case 1...")
        traj_M_x, traj_M_u = run_full_MPC(mpc_M, x_init)
        costs_M[sim_idx] = calculate_realized_cost(traj_M_x, traj_M_u)

        # Case 2
        print("Running Case 2...")
        traj_M_NN_x, traj_M_NN_u = run_full_MPC(mpc_M_NN, x_init)
        costs_M_NN[sim_idx] = calculate_realized_cost(traj_M_NN_x, traj_M_NN_u)

        # Case 3
        print("Running Case 3...")
        traj_N_plus_M_x, traj_N_plus_M_u = run_full_MPC(mpc_N_plus_M, x_init)
        costs_N_plus_M[sim_idx] = calculate_realized_cost(traj_N_plus_M_x, traj_N_plus_M_u)
        
    print(f"\nCompleted {N_sim} runs.\n\n")

    
    # --- Print Summary Statistics ---
    
    Mean_cost_M = np.mean(costs_M)
    Std_cost_M = np.std(costs_M)
    
    Mean_cost_M_NN = np.mean(costs_M_NN)
    Std_cost_M_NN = np.std(costs_M_NN)
    
    Mean_cost_N_plus_M = np.mean(costs_N_plus_M)
    Std_cost_N_plus_M = np.std(costs_N_plus_M)
    
    print("\n--- Summary Statistics over Multiple Simulations ---")
    
    print("="*80)
    print(f"{'SCENARIO':<35} | {'MEAN COST':<12} | {'STD DEV':<10}")
    print("-" * 80)
    print(f"{'1. M No Cost':<35} | {Mean_cost_M:12.2f} | {Std_cost_M:10.2f}")
    print(f"{'2. Short Horizon + NN Cost':<35} | {Mean_cost_M_NN:12.2f} | {Std_cost_M_NN:10.2f}")
    print(f"{'3. Benchmark (Long Horizon)':<35} | {Mean_cost_N_plus_M:12.2f} | {Std_cost_N_plus_M:10.2f}")
    print("="*80)
    
    # Plot 1: Cost Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    data_to_plot = [costs_M, costs_M_NN, costs_N_plus_M]
    ax.boxplot(data_to_plot, patch_artist=True, 
                        labels=['Short (No Cost)', 'Short + NN', 'Benchmark'])
    ax.set_title(f'Cost Distribution over {N_sim} Random Starts')
    ax.set_ylabel('Realized Cost (Lower is better)')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    plt.close('all')
    
if __name__ == "__main__":
    main()