import torch
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import os 
import warnings
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import time
import gc

# To ignore some warnings from libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from neural_network import NeuralNetwork
from adam.casadi.computations import KinDynComputations
from example_robot_data.robots_loader import load



"""
Model Predictive Control (MPC) Comparison for Double Pendulum.

This script implements and compares three different MPC strategies for the 
swing-up and balancing of a double pendulum system:
1. Short Horizon (M) without terminal cost.
2. Short Horizon (M) with a Learned Terminal Cost (Neural Network).
3. Long Horizon (N+M) as a performance benchmark.

Input: Trained PyTorch model (.pt) and dataset scaling parameters (.npz)
"""


# --- GLOBAL CONFIGURATION ---

# --- Global robotic model (pendulum) setup ---
print("\n--- Loading Robot ---")
robot = load("double_pendulum")
urdf_path = robot.urdf

nq = robot.model.nq  # Number of joint positions
nv = robot.model.nv  # Number of joint velocities
nu = 1;  # Number of control inputs (1 torque on first joint)

joint_names = [s for s in robot.model.names[1:]]

# ADAM function for the robot model
KinDyn = KinDynComputations(urdf_path, joint_names)

# 1. System Dimensions
STATE_DIM = 4    # [q1, q2, dq1, dq2]
CONTROL_DIM = 1  # [tau]
HIDDEN_SIZE = 256 # Neurons in hidden layers

# 2. Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
MODEL_PATH = os.path.join(SCRIPT_DIR, 'value_function_model_used.pt')
DATA_PATH = os.path.join(SCRIPT_DIR, 'pendulum_dataset_used.npz')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. Physical Parameters
DT = 0.05       # Time step [s]

# 4. MPC & Cost Weights
# Weights for state [q, dq] and control [u]
WEIGHTS_X = [4e0, 4e0, 5e-1, 1e-2]   # q1, q2, dq1, dq2
WEIGHTS_U = [1e0]                    # tau1
WEIGHT_FINAL = 1e-1                  # Terminal cost weight

# 5. Targets
TARGET_STATE = [0.0, 0.0, 0.0, 0.0] # Upright position
TARGET_CONTROL = [0.0]

# 6. Dynamics Function using Global Physics
# ---  Dynamics Definition (Using ADAM) ---
q_sym = ca.SX.sym('q', nq)  # State Symbolic
dq_sym = ca.SX.sym('dq', nv)
tau_sym = ca.SX.sym('tau', nu)

# Pendulum base frame (identity, zero velocity). This says that the base is fixed
H_b = ca.SX.eye(4)
v_b = ca.SX.zeros(6)

# Retrieve Matrices from ADAM
# Note: ADAM returns full matrices (base+joints), we slice [6:] to remove base part, and keep only joints stuff
mass_matrix_fun = KinDyn.mass_matrix_fun()
bias_force_fun = KinDyn.bias_force_fun()

M = mass_matrix_fun(H_b, q_sym)[6:, 6:]        # Inertia (2x2)
h = bias_force_fun(H_b, q_sym, v_b, dq_sym)[6:] # Coriolis/Gravity (2x1)

# Full torque vector (joint1 + joint2)
tau = ca.vertcat(tau_sym, 0.0) # [tau1, 0] (2x1)

# Compute Acceleration: ddq = inv(M) * (tau - h)
ddq_derived = ca.solve(M, tau - h)

# Define ODE system: x_dot = f(x, u)
x_sym = ca.vertcat(q_sym, dq_sym)           # State [q, dq]
u_sym = tau_sym                             # Control [tau]
x_dot = ca.vertcat(dq_sym, ddq_derived)     # Derivative [dq, ddq]


# Dynamics Integration (Euler Explicit) x_next = x_current + x_dot * dt
x_next_sym = x_sym + x_dot * DT

# CasADi Function for Dynamics
F = ca.Function('F', 
    [x_sym, u_sym], 
    [x_next_sym], 
    ['xk', 'uk'], 
    ['xf'])



# --- LOAD SCALING PARAMETERS ---
def get_scaling_parameters(filepath):
    """
    Loads the dataset, cleans the data, and calculates normalization factors.

    This function removes outliers to ensure dataset quality. 
    It defines the upper limits for states (X) and calculates the maximum 
    log-cost (Y) for neural network scaling and denormalization (Y was in log scale during the network training).

    Args:
        filepath (str): Path to the .npz file containing the dataset.

    Returns:
        X_MAX (np.ndarray): Array of upper bounds for the state [q, dq].
        Y_MAX (float): Maximum log-cost value for output denormalization.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at {filepath}")
    
    data = np.load(filepath)
    X = data['inputs']
    Y = data['targets']
     
    # Outlier Removal
    # 1. NaN removal
    # NB: technically there shouldn't be NaNs in the used dataset, but just in a previous one we had this issue
    nan_mask_X = np.isnan(X).any(axis=1) 
    nan_mask_Y = np.isnan(Y)             
    
    # Indices of valid samples (no NaN) 
    valid_mask = ~(nan_mask_X | nan_mask_Y)
    
    X_no_nan = X[valid_mask]
    Y_no_nan = Y[valid_mask]
    
    n_nan = len(X) - len(X_no_nan)
    if n_nan > 0:
        print(f"Removed {n_nan} samples containing NaN.")

    # 2. Outlier removal based on Y threshold
    threshold = 20000.0
    
    # Indices of samples below threshold
    outlier_mask = Y_no_nan <= threshold
    
    X_clean = X_no_nan[outlier_mask]
    Y_clean = Y_no_nan[outlier_mask]
    n_outliers = len(X_no_nan) - np.sum(outlier_mask)
    
    print(f"\n[OUTLIERS ANALYSIS]")
    print(f"Num of total samples: {len(Y)}")
    print(f"Samples above threshold ({threshold}): {n_outliers} ({n_outliers/len(Y)*100:.2f}%)")

    # NB: max values based on state limits
    X_MAX = np.array([np.pi, np.pi, 15.0, 6.0])
    
    Y_log = np.log(Y_clean + 1e-8)
    Y_MAX = np.max(Y_log) # log scaling to compress range
    
    print(f"\nData normalization factors: ")
    print(f"X max: {np.round(X_MAX, 3)}, Y max: {Y_MAX:.3f}")

    return X_MAX, Y_MAX


# --- CREATE TERMINAL COST FROM MODEL ---
def get_learned_terminal_cost(x_max_val, y_max_val):
    """
    Converts a trained PyTorch Value Function into a CasADi symbolic function.

    The function handles the internal loading of weights, input normalization 
    based on x_max_val, and output denormalization using y_max_val.

    Args:
        x_max_val (np.ndarray): Scaling array for input states [q, dq].
        y_max_val (float): Normalization factor for the log-scaled cost.

    Returns:
        ca.Function: A symbolic CasADi function representing the Learned Terminal Cost.
    """
    print("\n--- Converting PyTorch model to CasADi function ---")
    
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
    
    # Needed to do element-wise division for normalization (next code line)
    scale_factor = ca.DM(x_max_val.reshape(-1, 1))
    
    # Normalize inputs
    x_norm = x_sym / scale_factor
    
    # Get normalized output from NN
    y_norm = casadi_funct_form_nn(x_norm)
    
    # Denormalize output (return to linear scale)
    y_phys = np.exp(y_norm * y_max_val)
    
    return ca.Function('V_learned', [x_sym], [y_phys])


# --- MPC CLASS --- 
class MPC_Controller:
    """
    MPC Controller using CasADi Opti for pendulum swing-up and balance.
    
    This class handles the formulation of the Optimal Control Problem (OCP).
    
    Methods:
        solve_one_step: Returns the first optimal control action for a given state.
    """
    def __init__(self, horizon, dt=DT, use_terminal_cost=False, terminal_cost_func=None):
        """
        Initializes the MPC Optimization Problem.
        Sets up variables, physics constraints, and the IPOPT solver.
        
        Args:
        horizon (int): Number of steps in the prediction horizon.
        dt (float): Time step duration.
        use_terminal_cost (bool): If True, enables the Neural Network terminal cost.
        terminal_cost_func (ca.Function): The symbolic NN function for terminal cost.
        """
        
        #--- MPC Problem Setup ---
        self.N = horizon
        self.dt = dt
        self.use_terminal_cost = use_terminal_cost
        self.terminal_cost_func = terminal_cost_func
        self.opti = ca.Opti()
        
        # Variables & Parameters 
        self.X = self.opti.variable(STATE_DIM, self.N + 1) 
        self.U = self.opti.variable(CONTROL_DIM, self.N)
        
        self.X0_param = self.opti.parameter(STATE_DIM)        # Initial State
        self.X_targ_param = self.opti.parameter(STATE_DIM)    # Target State
        self.U_targ_param = self.opti.parameter(CONTROL_DIM)  # Target Control
        
        # Weights (Parameters)
        self.Wx_vect = self.opti.parameter(STATE_DIM) 
        self.Wu_vect = self.opti.parameter(CONTROL_DIM)
        self.Wf = self.opti.parameter(1)
        Wf = self.Wf
        
        # Warm-starting storage
        self.prev_X_sol = None
        self.prev_U_sol = None

        # - Cost Function & Constraints -
        J_running = 0
        J_final = 0
        
        # Weight Matrices from Vectors defined as parameters above
        Wx = ca.diag(self.Wx_vect)
        Wu = ca.diag(self.Wu_vect)

        # Initial constraint
        self.opti.subject_to(self.X[:, 0] == self.X0_param)
        
        # Cost computation over the horizon
        for k in range(self.N):
            x_k = self.X[:, k]
            u_k = self.U[:, k]

            # Cost accumulation
            err_x = x_k - self.X_targ_param
            err_u = u_k - self.U_targ_param
            J_running += err_x.T @ Wx @ err_x + err_u.T @ Wu @ err_u
            
            # Dynamics constraint 
            x_next = F(xk=x_k, uk=u_k)['xf']
            self.opti.subject_to(self.X[:, k+1] == x_next)

        # Terminal Cost
        if use_terminal_cost and terminal_cost_func is not None:
            
            cost_terminal = terminal_cost_func(self.X[:, self.N])
            J_final += Wf * cost_terminal

        # Total Cost to minimize
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
        """
        Computes the optimal control action for the current state.

        Updates the OCP parameters, applies a warm-start guess, and 
        invokes the IPOPT solver.

        Args:
            x_current (np.ndarray): The current state of the pendulum.
            target (np.ndarray, optional): Desired target state.

        Returns:
            float: The optimal torque to apply to the joint.
        """
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
        self.opti.set_value(self.Wf, WEIGHT_FINAL)
        
        # Warm-Starting
        # If previous solution exists, use it as initial guess
        # this takes the previous solution and shifts it by one time step
        if self.prev_X_sol is not None:
            x_guess = np.hstack((self.prev_X_sol[:, 1:], self.prev_X_sol[:, -1:]))
            u_guess = np.hstack((self.prev_U_sol[:, 1:], self.prev_U_sol[:, -1:]))
            self.opti.set_initial(self.X, x_guess)
            self.opti.set_initial(self.U, u_guess)
        
        try:
            # Solve the OCP
            sol = self.opti.solve()
            
            x_val = sol.value(self.X)
            u_val = sol.value(self.U)

            # Debug: Print terminal state and NN cost
            #x_terminal = x_val[:, -1] 
            #print("-" * 30)
            #print(f"MPC Step - Input alla NN (x_terminal):")
            #print(f"  q1: {x_terminal[0]:.4f}, q2: {x_terminal[1]:.4f}")
            #print(f"  dq1: {x_terminal[2]:.4f}, dq2: {x_terminal[3]:.4f}")
            # Se vuoi vedere il costo che la NN calcolerà:
            #v_raw_numeric = self.terminal_cost_func(x_terminal).full().flatten()[0]
            #print(f"  V_nn (Raw): {v_raw_numeric:.4f}, V_nn (Used): {v_raw_numeric:.4f}")
            #print("-" * 30)
            
            # To solve an error that occurred
            if u_val.ndim == 1: u_val = u_val.reshape(CONTROL_DIM, -1)
            if x_val.ndim == 1: x_val = x_val.reshape(STATE_DIM, -1)

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
            if u_debug.ndim == 1: u_debug = u_debug.reshape(CONTROL_DIM, -1)
            if x_debug.ndim == 1: x_debug = x_debug.reshape(STATE_DIM, -1)
            
            # Store for warm start
            self.prev_X_sol = x_debug
            self.prev_U_sol = u_debug
            
            # Return the calculated control action (even if sub-optimal)
            return float(u_debug[0, 0])


# --- MPC SIMULATION ---
def run_full_MPC(controller, x0, steps=100):
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
    
    for _ in range(steps):
        u = controller.solve_one_step(curr_x) # Get first control action
        traj_u.append(u)
        
        # Apply dynamics with the function at the beginning of the script 
        x_next = F(xk=curr_x, uk=u)
        curr_x = x_next['xf'].full().flatten()

        traj_x.append(curr_x)
        
    return np.array(traj_x), np.array(traj_u)


# --- ACTUAL COST FOR COMPARISON ---
def calculate_realized_cost(traj_x, traj_u, threshold=0.1):
    """
    Calculates the total cost accumulated during the real simulation.
    WARNING: the cost computed by this function is NOT the same as the one minimized by the MPC.
    
    Args:
        traj_x: np.ndarray of shape (T, STATE_DIM)
        traj_u: np.ndarray of shape (T, CONTROL_DIM)
        threshold (float): Convergence radius around the target.
        
    Returns:
        total_cost (float): Sum of quadratic stage costs.
        time_to_target (float): Time (seconds) to reach the threshold. 
                                Returns -1 if never reached.
    """
    
    x_target = np.array(TARGET_STATE)
    u_target = np.array(TARGET_CONTROL)
    
    # Use Global Weights
    Wx = np.diag(WEIGHTS_X) 
    Wu = np.diag(WEIGHTS_U)    
    
    total_cost = 0.0
    time_to_target = -1.0
    n_steps = len(traj_u)
    
    for k in range(n_steps):
        x_k = traj_x[k] 
        u_k = traj_u[k]
        
        err_x = x_k - x_target
        err_u = u_k - u_target

        time_weight = (k + 1) / n_steps  # Increasing weight over time (optional)

        step_cost = np.sqrt(err_x.T @ Wx @ err_x) + np.sqrt(err_u.T @ Wu @ err_u)
        total_cost += time_weight * step_cost
        
        if time_to_target < 0:
            dist = np.linalg.norm(err_x) # Euclidean distance to target
            if dist < threshold:
                time_to_target = k * DT # Time in seconds
        
    return float(total_cost), float(time_to_target)

# --- PENDULUM ANIMATION FUNCTION ---
def animate_three_pendulums(traj_list, titles):
    """
    Visualizes 3 pendulums in parallel on Meshcat by creating a composite model.
    Includes replay functionality and Color Coding.
    """
    print("--- Preparing Meshcat Animation ---")
    
    # 1. Load base model (single robot)
    robot_single = load("double_pendulum")
    rmodel_single = robot_single.model
    vmodel_single = robot_single.visual_model
    
    # 2. Create empty models for the composite system
    rmodel_full = pin.Model()
    vmodel_full = pin.GeometryModel()
    
    # 3. Define offsets and Colors
    offsets = [-0.5, 0.0, 0.5] 
    
    # Robot colors (RGBA): Rosso, Blu, Verde
    colors = [
        np.array([1.0, 0.2, 0.2, 1.0]), # Red 
        np.array([0.2, 0.2, 1.0, 1.0]), # Blue  
        np.array([0.2, 1.0, 0.2, 1.0])  # Green 
    ]

    # 4. Merge the 3 robots into one model
    
    for i, offset in enumerate(offsets):
        placement = pin.SE3(np.eye(3), np.array([0, offset, 0]))
        
        # Copy models
        model_to_add = rmodel_single.copy()
        vmodel_to_add = vmodel_single.copy()
        
        suffix = f"_{i}" # e.g., "_0", "_1", "_2"
        
        # Rename Frames
        for frame in model_to_add.frames:
            frame.name += suffix
            
        # Rename Joints
        for k, name in enumerate(model_to_add.names):
            if name != "universe":
                model_to_add.names[k] = name + suffix

        # Rename Geometries and Apply Color
        for geom in vmodel_to_add.geometryObjects:
            geom.name += suffix
            geom.meshColor = colors[i] 

        # Append the renamed model
        rmodel_full, vmodel_full = pin.appendModel(
            rmodel_full,        # Destination Model
            model_to_add,       # Source Model
            vmodel_full,        # Destination Geometry
            vmodel_to_add,      # Source Geometry
            0,                  # Parent Joint (0 = universe)
            placement           # Relative placement
        )

    # 5. Initialize Visualizer
    viz = MeshcatVisualizer(rmodel_full, vmodel_full, vmodel_full)
    
    # Start Meshcat Server
    try:
        viz.initViewer(open=True)
        viz.loadViewerModel()
        time.sleep(1.0) 
    except ImportError:
        print("Error: Meshcat not installed or connection failed.")
        return

    # Print legend console
    print("\n" + "="*40)
    print(" VISUALIZATION LEGEND ")
    print("="*40)
    color_names = ["RED", "BLUE", "GREEN"]
    for i, title in enumerate(titles):
        clean_title = title.replace('\n', ', ')
        print(f"Robot {color_names[i]}: {clean_title}")
    print("="*40 + "\n")

    # 6. Animation Loop with Replay
    max_steps = max([len(t) for t in traj_list])
    dt_sleep = 0.05 
    
    while True:
        # Reset visualizer
        q_start_combined = []
        for i in range(3):
            q_start_combined.append(traj_list[i][0, :2])
        viz.display(np.concatenate(q_start_combined))
        
        input("Press Enter to start the animation...") 

        for t in range(max_steps):
            q_combined = []
            
            for i in range(3):
                traj = traj_list[i]
                # Take last valid state
                idx = min(t, len(traj) - 1)
                q_robot = traj[idx, :2] 
                q_combined.append(q_robot)
                
            q_viz = np.concatenate(q_combined)
            viz.display(q_viz)
            time.sleep(dt_sleep)
            
        print("Animation finished.")
        
        # Replay logic
        user_choice = input("Do you want to replay? (y/n): ").strip().lower()
        if user_choice != 'y':
            print("Closing visualization.")
            break
    
    

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
    
    #Debug: Check value function at target state
    #val_at_zero = terminal_cost_from_nn([0.0, 0.0, 0.0, 0.0])
    #print(f"DEBUG: Valore della NN al target [0,0,0,0]: {val_at_zero}")
    
    if terminal_cost_from_nn is None:
        return

    print("\n--- Initializing Controllers ---")
    
    M = 14   # Short horizon 
    N = 100 # Long horizon 
    
    time_convergence_tresh = 0.08 # Threshold for convergence time calculation
    
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
    # Can be changed to any other initial state
    #x_init = np.array([ 2.88, -0.89,  0.0   ,  0.0   ]) 
    x_init = np.array([ np.pi, 0.0,  0.0   ,  0.0   ]) 
    #x_init = np.array([1.14152082, 2.82742641, 0.5, -0.9999466 ])
    #x_init = np.array([-2, -0.60, 0.3, -0.83 ]) 
    #x_init = np.array([0.1, -2.60, 0.6, -0.23 ])    
    #x_init = np.array([-0.82, -0.60, 0.99, -0.83 ])


    # Set random initial state
    #q1_rand = np.random.uniform(-np.pi, np.pi) # q1 rad
    #q2_rand = np.random.uniform(-np.pi*0.8, np.pi*0.8)  # q2 rad
    #q_rand = np.array([q1_rand, q2_rand])
    #dq_rand = np.random.uniform(-1.0, 1.0, size=nv) # [dq1, dq2] rad/s
    #x_init = np.concatenate([q_rand, dq_rand])
        
    print("\n--- Starting Simulations & Calculating Costs for 1 Example Initial State ---")
    print(f"\nExample Initial State: {x_init.round(3)}\n")
    
# --- SIM MPC1 ---
    print("Running Case 1...")
    traj_M_x, traj_M_u = run_full_MPC(mpc_M, x_init)
    cost_M_ex, time_M_ex = calculate_realized_cost(traj_M_x, traj_M_u, threshold=time_convergence_tresh)

    # --- SIM MPC2 ---
    print("Running Case 2...")
    traj_M_NN_x, traj_M_NN_u = run_full_MPC(mpc_M_NN, x_init)
    cost_M_NN_ex, time_M_NN_ex = calculate_realized_cost(traj_M_NN_x, traj_M_NN_u, threshold=time_convergence_tresh)

    # --- SIM MPC3 ---
    print("Running Case 3...")
    traj_N_plus_M_x, traj_N_plus_M_u = run_full_MPC(mpc_N_plus_M, x_init)
    cost_N_plus_M_ex, time_N_plus_M_ex = calculate_realized_cost(traj_N_plus_M_x, traj_N_plus_M_u, threshold=time_convergence_tresh)
    
    print("\n[NOTE]: time = -1 means the target was not reached within the simulation time, convergence treshold = {time_convergence_tresh}")

    # --- Print Comparison Table ---
    print("\n" + "="*80)
    print(f"{'SCENARIO':<35} | {'REALIZED COST':<15} | {'SETTLING TIME (s)':<15}")
    print("-" * 80)
    print(f"{'Case 1: Short Horizon (No Cost)':<35} | {cost_M_ex:15.2f} | {time_M_ex:.2f}")
    print(f"{'Case 2: Short Horizon + NN Cost':<35} | {cost_M_NN_ex:15.2f} | {time_M_NN_ex:.2f}")
    print(f"{'Case 3: Long Horizon (Benchmark)':<35} | {cost_N_plus_M_ex:15.2f} | {time_N_plus_M_ex:.2f}")
    print("="*80 + "\n")

    # --- Plot Results of the Example (Positions and Velocities) ---
    print("--- Plotting Results of the example ---")

    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    
    # --- Row 1: Positions ---
    
    # Subplot (0,0): q1
    ax[0, 0].plot(traj_M_x[:, 0], 'r--', label=f'M no Cost (J={cost_M_ex:.1f})')
    ax[0, 0].plot(traj_N_plus_M_x[:, 0], 'g-', alpha=0.5, linewidth=3, label=f'Benchmark (J={cost_N_plus_M_ex:.1f})')
    ax[0, 0].plot(traj_M_NN_x[:, 0], 'b.-', label=f'M + NN Cost (J={cost_M_NN_ex:.1f})')
    ax[0, 0].axhline(TARGET_STATE[0], color='k', linestyle=':', label='Target')
    ax[0, 0].set_title("q1")
    ax[0, 0].set_ylabel("Angle [rad]")
    ax[0, 0].legend()
    ax[0, 0].grid(True)

    # Subplot (0,1): q2
    ax[0, 1].plot(traj_M_x[:, 1], 'r--', label='M no Cost')
    ax[0, 1].plot(traj_N_plus_M_x[:, 1], 'g-', alpha=0.5, linewidth=3, label='Benchmark')
    ax[0, 1].plot(traj_M_NN_x[:, 1], 'b.-', label='M + NN Cost')
    ax[0, 1].axhline(TARGET_STATE[1], color='k', linestyle=':', label='Target')
    ax[0, 1].set_title("q2")
    ax[0, 1].set_ylabel("Angle [rad]")
    ax[0, 1].legend()
    ax[0, 1].grid(True)

    # --- Row 2: Velocities ---

    # Subplot (1,0): dq1
    ax[1, 0].plot(traj_M_x[:, 2], 'r--', label='M no Cost')
    ax[1, 0].plot(traj_N_plus_M_x[:, 2], 'g-', alpha=0.5, linewidth=3, label='Benchmark')
    ax[1, 0].plot(traj_M_NN_x[:, 2], 'b.-', label='M + NN Cost')
    ax[1, 0].axhline(TARGET_STATE[2], color='k', linestyle=':', label='Target')
    ax[1, 0].set_title("dq1")
    ax[1, 0].set_xlabel("Time Steps")
    ax[1, 0].set_ylabel("Vel [rad/s]")
    ax[1, 0].legend()
    ax[1, 0].grid(True)

    # Subplot (1,1): dq2
    ax[1, 1].plot(traj_M_x[:, 3], 'r--', label='M no Cost')
    ax[1, 1].plot(traj_N_plus_M_x[:, 3], 'g-', alpha=0.5, linewidth=3, label='Benchmark')
    ax[1, 1].plot(traj_M_NN_x[:, 3], 'b.-', label='M + NN Cost')
    ax[1, 1].axhline(TARGET_STATE[3], color='k', linestyle=':', label='Target')
    ax[1, 1].set_title("dq2")
    ax[1, 1].set_xlabel("Time Steps")
    ax[1, 1].set_ylabel("Vel [rad/s]")
    ax[1, 1].legend()
    ax[1, 1].grid(True)

    plt.suptitle(f"MPCs Comparison - Init State: {x_init.round(2)}")
    plt.tight_layout()
    plt.show()
    
    # --- Pendulums Animation ---
    print("\n--- Showing Animation ---")
    
    trajectories = [traj_M_x, traj_M_NN_x, traj_N_plus_M_x]
    
    titles = [
        f"1. M No Cost\nJ = {cost_M_ex:.1f}",
        f"2. M+NN Cost \nJ = {cost_M_NN_ex:.1f}",
        f"3. N+M (Benchmark) \nJ = {cost_N_plus_M_ex:.1f}"
    ]
    
    animate_three_pendulums(trajectories, titles)
    
    # Clean up of the MPCs to avoid solving issues in next iteration
    # NB we are not sure if really needed 
    del mpc_M
    del mpc_M_NN
    del mpc_N_plus_M
    gc.collect()
    
    
    # --- Simulations over multiple random initial states ---
    print("\n--- Running Simulations over Multiple Random Initial States ---")
    N_sim = 30
    
    # Initialize costs storage
    costs_M = np.zeros(N_sim)
    costs_M_NN = np.zeros(N_sim)
    costs_N_plus_M = np.zeros(N_sim)
    times_M = np.zeros(N_sim)
    times_M_NN = np.zeros(N_sim)
    times_N_plus_M = np.zeros(N_sim)

    for sim_idx in range(N_sim):
        
        # Re-initialize MPC controllers to avoid old solutions issues
        # NB we are not sure if really needed
        mpc_M = MPC_Controller(horizon=M, use_terminal_cost=False)
        mpc_M_NN = MPC_Controller(horizon=M, use_terminal_cost=True, terminal_cost_func=terminal_cost_from_nn)
        mpc_N_plus_M = MPC_Controller(horizon=N+M, use_terminal_cost=False)

        
        # Set random initial state
        q1_rand = np.random.uniform(-np.pi, np.pi) # q1 rad
        # To avoid singularities near upright position for q2 i do pi*0.8
        q2_rand = np.random.uniform(-np.pi*0.8, np.pi*0.8)  # q2 rad
        q_rand = np.array([q1_rand, q2_rand])
        dq_rand = np.random.uniform(-1.0, 1.0, size=nv) # [dq1, dq2] rad/s
        x_init = np.concatenate([q_rand, dq_rand])
        
        #x_init = np.array([-0.80, -2.34, -0.37, 0.28])
        
         # String [q1, q2, dq1, dq2]
        state_str = (f"q: [{x_init[0]:.6f}, {x_init[1]:.6f}] rad, "
                     f"dq: [{x_init[2]:.6f}, {x_init[3]:.6f}] rad/s")
        
        # Print initial state and current simulation number
        print(f"\nRun {sim_idx+1}/{N_sim}")
        print(f"Init State -> {state_str}")
    

        # Case 1
        print("Running Case 1...")
        traj_M_x, traj_M_u = run_full_MPC(mpc_M, x_init)
        costs_M[sim_idx], times_M[sim_idx] = calculate_realized_cost(traj_M_x, traj_M_u, threshold=time_convergence_tresh)
        print(f"  [Case 1] Cost: {costs_M[sim_idx]:.3f} | Time: {times_M[sim_idx]:.2f}s")

        # Case 2
        print("Running Case 2...")
        traj_M_NN_x, traj_M_NN_u = run_full_MPC(mpc_M_NN, x_init)
        costs_M_NN[sim_idx], times_M_NN[sim_idx] = calculate_realized_cost(traj_M_NN_x, traj_M_NN_u, threshold=time_convergence_tresh)
        print(f"  [Case 2] Cost: {costs_M_NN[sim_idx]:.2f} | Time: {times_M_NN[sim_idx]:.2f}s")
        
        # Case 3
        print("Running Case 3...")
        traj_N_plus_M_x, traj_N_plus_M_u = run_full_MPC(mpc_N_plus_M, x_init)
        costs_N_plus_M[sim_idx], times_N_plus_M[sim_idx] = calculate_realized_cost(traj_N_plus_M_x, traj_N_plus_M_u, threshold=time_convergence_tresh)
        print(f"  [Case 3] Cost: {costs_N_plus_M[sim_idx]:.3f} | Time: {times_N_plus_M[sim_idx]:.2f}s")
        
        # Clean up of the MPCs to avoid solving issues in next iteration
        # NB not sure if really needed 
        del mpc_M
        del mpc_M_NN
        del mpc_N_plus_M
        gc.collect()
        
        
    print(f"\nCompleted {N_sim} runs.\n\n")

    # --- Print Summary Statistics ---
    def get_stats(c_array, t_array):
        valid_t = t_array[t_array > 0]
        m_cost = np.mean(c_array)
        m_time = np.mean(valid_t) if len(valid_t) > 0 else 0.0
        success_rate = (len(valid_t) / len(t_array)) * 100
        return m_cost, m_time, success_rate

    m_c1, m_t1, sr1 = get_stats(costs_M, times_M)
    m_c2, m_t2, sr2 = get_stats(costs_M_NN, times_M_NN)
    m_c3, m_t3, sr3 = get_stats(costs_N_plus_M, times_N_plus_M)

    print("\n--- Summary Statistics over Multiple Simulations ---")
    print("="*95)
    print(f"{'SCENARIO':<35} | {'MEAN COST':<12} | {'MEAN TIME (s)':<15} | {'SUCCESS'}")
    print("-" * 95)
    print(f"{'1. M No Cost':<35} | {m_c1:12.2f} | {m_t1:15.2f} | {sr1:6.1f}%")
    print(f"{'2. Short Horizon + NN Cost':<35} | {m_c2:12.2f} | {m_t2:15.2f} | {sr2:6.1f}%")
    print(f"{'3. Benchmark (Long Horizon)':<35} | {m_c3:12.2f} | {m_t3:15.2f} | {sr3:6.1f}%")
    print("="*95)
    
    
if __name__ == "__main__":
    main()