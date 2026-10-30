import os
import time
import numpy as np
import casadi as ca
import multiprocessing as mp

from tqdm import tqdm
from adam.casadi.computations import KinDynComputations
from example_robot_data.robots_loader import load

"""
Optimal Control Problem (OCP) Solver & Dataset Generator for Double Pendulum Swing-up.

This module implements a trajectory optimization pipeline using CasADi and the ADAM  library for rigid body dynamics. It generates a dataset mapping initial  states to optimal costs via parallelized IPOPT solvers.

Output: Saves a .npz file containing 'inputs' (states) and 'targets' (optimal costs).
"""



# --- Global Constants ---
N_HORIZON = 100  # Prediction Horizon
DT = 0.05       # Time step duration (s)
X_TARGET = np.array([0.0, 0.0, 0.0, 0.0])  # Target State: [q1, q2, dq1, dq2] (upright position)

# Global dictionary to hold worker-specific data (for multiprocessing)
worker_data = {}

# --- Global robotic model (pendulum) setup ---
print("--- Loading Robot ---")
robot = load("double_pendulum")
urdf_path = robot.urdf

nq = robot.model.nq  # Number of joint positions
nv = robot.model.nv  # Number of joint velocities

joint_names = [s for s in robot.model.names[1:]]

# KinDyn Computations for the double pendulum
KinDyn_pend = KinDynComputations(urdf_path, joint_names)



# --- OCP Solver Creation Function ---
def create_solver(KinDyn, N, dt):
    """
    Constructs the Optimal Control Problem (OCP) using CasADi.

    Args:
        KinDyn (KinDynComputations): Dynamics function via ADAM.
        N (int): The prediction horizon (number of discrete time steps).
        dt (float): The time step duration in seconds (integration interval).
    
    Returns:
        tuple: A tuple containing:
            - opti (casadi.Opti): The NLP solver instance containing the optimization problem.
            - params: Symbolic parameters for setting boundary conditions (mutable via set_value).
            - variables: Symbolic expressions used to extract numerical results (trajectories, cost and controls) after solving.
        More details in the function body end.
    """
    # CasADi Opti Instance
    opti = ca.Opti()    

    # --- Dimensions ---
    nx = nq + nv  # State: [q1, q2, dq1, dq2] (Angles, Angular Velocities)
    nu = 1  # Controls: [tau1] (Torque at first joint only)
    

    # --- Variables & Parameters ---
    X = opti.variable(nx, N+1)         # nx rows, N+1 columns
    U = opti.variable(nu, N)           # nu rows, N columns
    X0_param = opti.parameter(nx)      # Initial State
    X_targ_param = opti.parameter(nx)  # Target State
    U_targ_param = opti.parameter(nu)  # Target Control

    # ---  Dynamics Definition (Using ADAM) ---
    q_sym = ca.SX.sym('q', nq)      # Position Symbolic [q1, q2]
    dq_sym = ca.SX.sym('dq', nv)    # Velocity Symbolic [dq1, dq2]
    tau_sym = ca.SX.sym('tau', nu)  # Control Symbolic [tau1]
    
   # Pendulum base frame (identity, zero velocity). This says that the base is fixed
    H_b = ca.SX.eye(4)
    v_b = ca.SX.zeros(6)
    
    # Retrieve Matrices from ADAM
    # Note: ADAM returns full matrices (base+joints), we slice [6:] to remove base part, and keep only joints stuff
    mass_matrix_fun = KinDyn.mass_matrix_fun()
    bias_force_fun = KinDyn.bias_force_fun()

    M = mass_matrix_fun(H_b, q_sym)[6:, 6:]          # Inertia (2x2)
    h = bias_force_fun(H_b, q_sym, v_b, dq_sym)[6:]  # Coriolis/Gravity (2x1)

    # Full torque vector (joint1 + joint2)
    tau = ca.vertcat(tau_sym, 0.0) # [tau1, 0] (2x1)
    
    # Compute Acceleration: ddq = inv(M) * (tau - h)
    ddq = ca.solve(M, tau - h)
    
    # Define ODE system: x_dot = f(x, u)
    x_sym = ca.vertcat(q_sym, dq_sym)  # State [q1, q2, dq1, dq2]
    u_sym = tau_sym                    # Control [tau1]
    x_dot = ca.vertcat(dq_sym, ddq)    # State Derivative [dq1, dq2, ddq1, ddq2]


    # Dynamics Integration (Euler Explicit) x_next = x_current + x_dot * dt
    x_next_sym = x_sym + x_dot * dt
    
    # CasADi Function for Dynamics (computes x_next given x and u)
    F = ca.Function(
        'F',            # Function name
        [x_sym, u_sym], # Input symbols
        [x_next_sym],   # Output symbols
        ['xk', 'uk'],   # Input names
        ['xf']          # Output names
    )

    # --- Cost Function & Constraints ---
    J = 0

    # Weights Matrices
    weights_x = [4e0, 4e0, 5e-1, 1e-2] # q1, q2, dq1, dq2
    weights_u = [1e0]                    # tau1
    
    Wx = ca.diag(weights_x) 
    Wu = ca.diag(weights_u) 

    # Loop over the Horizon to build the Cost Function
    for k in range(N):
        x_k = X[:, k]
        u_k = U[:, k]
        
        # Cost accumulation
        err_x = x_k - X_targ_param
        err_u = u_k - U_targ_param
        J += err_x.T @ Wx @ err_x + err_u.T @ Wu @ err_u
        
        # Dynamics constraint 
        x_next = F(xk=x_k, uk=u_k)['xf']
        opti.subject_to(X[:, k+1] == x_next)

    # Initial condition constraint
    opti.subject_to(X[:, 0] == X0_param)
    
    # Objective = Minimize Cost J
    opti.minimize(J)
    
# --- Solver Options ---
    p_opts = {
        'expand': False,   
        'print_time': False  # Don't print internal CasADi timing stats
    } 
    
    s_opts = {
        'max_iter': 1000,    
        'print_level': 0,    # 0 = suppress iteration logs
        'sb': 'yes',         # Hides the initial IPOPT header info
        'tol': 1e-4          # Accuracy tolerance for convergence
    }
    
    # Solver configuration
    opti.solver('ipopt', #Use Interior Point OPTimizer 
                p_opts, 
                s_opts
    )
    
    # --- Return Values ---
    # Returns:
    # 1. opti: The NLP solver instance containing the optimization problem.
    # 2. params: Symbolic parameters for setting boundary conditions (mutable via set_value).
    # 3. variables: Symbolic expressions used to extract numerical results (trajectories, cost) after solving.
    # NOTE: Structured like this for clarity and orderliness. One could just return everything in sequence instead of this symbolic tuple.
    return opti, \
           {'X0': X0_param, 'X_targ': X_targ_param, 'U_targ': U_targ_param}, \
           {'J': J, 'X': X, 'U': U}


def init_worker():
    """ 
    Initializes the robot and solver ONCE per CPU core.
    
    This function avoids wasting time. Instead of rebuilding the solver 
    for every new calculation, it builds it once per core and keeps it 
    ready for use.
    """
    from example_robot_data.robots_loader import load
    from adam.casadi.computations import KinDynComputations
    
    robot_local = load("double_pendulum")
    joint_names = [s for s in robot_local.model.names[1:]]
    KinDyn_local = KinDynComputations(robot_local.urdf, joint_names)
    
    # Pre-build the solver
    opti, params, variables = create_solver(KinDyn_local, N_HORIZON, DT)
    
    # Store them in the global dictionary of the child process (worker_data is a global variable)
    worker_data['opti'] = opti
    worker_data['params'] = params
    worker_data['vars'] = variables
    
    
# --- Single OCP Solve Function for Multiprocessing ---
def solve_single_ocp(x_init):
    """
    Worker function for parallel processing. 
    Each CPU core executes this function independently: it loads a local instance of the robot model, builds the OCP solver, and attempts to find the optimal trajectory for a specific initial state.
    
    Args:
        x_init (np.ndarray): The initial state [q1, q2, dq1, dq2] to solve for.
        
    Returns:
        tuple: (x_init, optimal_cost, success_flag)
    """
    
    # Retrieve the pre-built solver and parameters from the global variable worker_data
    opti = worker_data['opti']
    params = worker_data['params']
    variables = worker_data['vars']
    
    # Set initial guesses to zero to clear any previous solution
    opti.set_initial(worker_data['vars']['X'], np.tile(X_TARGET.reshape(-1,1), (1, N_HORIZON+1)))
    opti.set_initial(worker_data['vars']['U'], 0.0)
    
    # Reset parameters for the new x_init
    opti.set_value(params['X_targ'], X_TARGET)
    opti.set_value(params['U_targ'], [0.0])
    opti.set_value(params['X0'], x_init)
    
    try:
        sol = opti.solve()
        return x_init, sol.value(variables['J']), True
    except:
        return x_init, None, False
    
    

# --- Dataset Generation Function in Parallel ---
def generate_dataset(N_samples):
    """
    Generates a dataset of optimal trajectories by solving the OCP for random initial states.
    Saves the inputs (state) and targets (optimal cost) to a file.
    
    So the file created will contain 2 arrays:
        inputs:  Nx4 array of states [q, dq]
        targets: Nx1 array of optimal costs (J)
    
    To load the dataset, use:
        data = np.load(DATA_PATH)
        X = data['inputs']   
        Y = data['targets']  
    
    Args:
        None 
    """
    
    print(f"\n==================================================")
    print(f" DATASET GENERATION FOR DOUBLE PENDULUM SWING-UPS")
    print(f"==================================================")
    print(f"\n--- Starting Parallel Generation on {mp.cpu_count()} cores ---")

   # Initialize random initial states
    all_x_init = []
    for _ in range(N_samples):
        q1_rand = np.random.uniform(-np.pi, np.pi)              #rad
        q2_rand = np.random.uniform(-np.pi*0.90, np.pi*0.90)    #rad
        dq1_rand = np.random.uniform(-8.0, 8.0)                 #rad/s
        dq2_rand = np.random.uniform(-15.0, 15.0)               #rad/s
        all_x_init.append(np.concatenate([[q1_rand, q2_rand], [dq1_rand, dq2_rand]]))

    # Initialize storage lists and counters
    inputs = []
    targets = []
    success_count = 0
    fail_count = 0

    # Parallel OCP Solving
    with mp.Pool(processes=mp.cpu_count(), initializer=init_worker) as pool:
        
        # Progress Bar Setup
        pbar = tqdm(pool.imap_unordered(solve_single_ocp, all_x_init), 
                    total=N_samples, 
                    desc="Solving OCPs")
        
        for x_init, cost, success in pbar:
            if success:
                inputs.append(x_init)
                targets.append(cost)
                success_count += 1
            else:
                fail_count += 1
                # Print failed initial states for debugging
                tqdm.write(f" [!] Solver Failed for x_init: {np.round(x_init, 2)}")

            # Status Update
            success_rate = (success_count / (success_count + fail_count)) * 100
            pbar.set_postfix({'Costs': f"{cost:.2f}" if cost is not None else "N/A",
                'Fails': fail_count, 
                'Succ. Rate': f"{success_rate:.2f}%"
            })

    # Save Dataset to File
    X_data = np.array(inputs)
    Y_data = np.array(targets)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'pendulum_dataset.npz')
    np.savez(filename, inputs=X_data, targets=Y_data)
    
    print(f"\nDataset created. Final Success Rate: {success_count/N_samples*100:.2f}%")
    return X_data, Y_data



# --- OCP Solve and Visualize Function ---
def solve_and_visualize(x_start_val):
    """
    Solves the Optimal Control Problem (OCP) for a given initial state and visualizes the resulting trajectory using Meshcat.

    Args:
        x_start_val (np.ndarray): The initial state vector [q, dq] to start the simulation from.
    """
    print(f"\n==================================================")
    print(f" EXAMPLE VISUALIZATION: OCP from state: {np.round(x_start_val, 2)}")
    print(f"==================================================")

    # --- Simulation Parameters ---
    N_horizon = N_HORIZON  
    dt = DT       
    
    # Create a fresh solver instance
    opti, params, vars = create_solver(KinDyn_pend, N_horizon, dt)
    
    # Set parameters (Initial State and Targets)
    opti.set_value(params['X0'], x_start_val)
    opti.set_value(params['X_targ'], X_TARGET)
    opti.set_value(params['U_targ'], [0.0])
    
    # --- Solve ---
    try:
        sol = opti.solve()
        
        # Print exit status
        status = sol.stats()['return_status']
        print(f"Solver Status: {status}")
        
        # Extract optimal values
        X_opt = sol.value(vars['X'])
        J_opt = sol.value(vars['J'])
        print(f"SOLUTION FOUND! Cost: {J_opt:.4f}")
        
    except Exception as e:
        print(f" SOLVER FAILED: {e}")
        # Retrieve the partial solution for debugging purposes
        try:
            X_opt = opti.debug.value(vars['X'])
        except:
            return

    # --- Visualization ---
    if not hasattr(robot, 'viz') or robot.viz is None:
        try:
            robot.initViewer(loadModel=True)
        except:
            print(" Meshcat not found.")
            return

    # Display the initial configuration (t=0) before the animation starts
    q0 = X_opt[:nq, 0]
    robot.display(q0)
    
    N_steps = X_opt.shape[1]
    
    while True:
        input(f">>> Press ENTER to animate ({N_steps} steps)...")
        
        for k in range(N_steps):
            q_k = X_opt[:nq, k]
            robot.display(q_k)
            # Small delay to match real-time execution
            time.sleep(dt) 
            
        robot.display(q0) 
            
        if input("   Replay? (y/n): ").lower() == 'n': break
    
    
    
# --- Main Execution ---
def main():

    #Example visualizations of OCP solutions
    
    x_ex = np.array([np.pi/2, np.pi/1.7, 0.0, 0.0]) 
    #x_ex = np.array([ 2.88, -0.89,  0.   ,  0.   ])
    #x_ex = np.array([ 0.47, -0.47, 0.0,  1.5])
    #x_ex = np.array([-1.2, 0.14, 1.65, 0.53])
    #x_ex = np.array([ 2.88, 0.89, -10.0,  -10.0])


    # Random initial state example
    #q1_rand = np.random.uniform(-np.pi, np.pi)
    #q2_rand = np.random.uniform(-np.pi*0.90, np.pi*0.90)
    #dq1_rand = np.random.uniform(-6.0, 6.0) 
    #dq2_rand = np.random.uniform(-15.0, 15.0) 
    #x_ex = np.concatenate([[q1_rand, q2_rand], [dq1_rand, dq2_rand]])

    
    init_worker() 
    
    xin, cost, flag = solve_single_ocp(x_ex)
    
    print(f"Single OCP Solve Result - Init: {np.round(xin,2)}, Cost: {cost}, Success: {flag}")
    
    # Visualize the example solution
    solve_and_visualize(x_ex)

    # Generate dataset
    generate_dataset(N_samples=10000)


if __name__ == "__main__":
    main()
