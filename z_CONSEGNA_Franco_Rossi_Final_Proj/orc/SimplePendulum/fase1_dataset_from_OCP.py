import os
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.pyplot as plt



def create_teacher_solver(N, dt):
    """
    Constructs the Optimal Control Problem (OCP) using CasADi.
    Defines the pendulum dynamics, cost function, and constraints to act as the 'Teacher'.

    Args:
        N (int): The prediction horizon (number of discrete time steps).
        dt (float): The time step duration in seconds (integration interval).
    
    Returns:
        tuple: A tuple containing:
            - opti (casadi.Opti): The NLP solver instance containing the optimization problem.
            - params: Symbolic parameters for setting boundary conditions and weights (mutable via set_value).
            - variables: Symbolic expressions used to extract numerical results (trajectories, cost) after solving.
        More details in the function body end.
    """
    # CasADi Opti Instance
    opti = ca.Opti()

    # --- Dimensions ---
    n_x = 2  # State: [q, dq] (Angle, Angular Velocity)
    n_u = 1  # Control: [tau] (Torque)

    # --- Variables & Parameters ---
    X = opti.variable(n_x, N+1)  # n_x rows, N+1 columns
    U = opti.variable(n_u, N)
    
    X0_param = opti.parameter(n_x)      # Initial State
    X_targ_param = opti.parameter(n_x)  # Target State
    U_targ_param = opti.parameter(n_u)  # Target Control
    Wx_vect = opti.parameter(n_x)       # State Weights vector
    Wu_vect = opti.parameter(n_u)       # Control Weights vector

    # --- Physics (Pendulum Dynamics) ---
    m, l, g = 1.0, 1.0, 9.81 # M = 1kg, L = 1m, g = 9.81m/s²

    # Symbolic Variables definition
    x_sym = ca.MX.sym('x', n_x)
    u_sym = ca.MX.sym('u', n_u)
    q, dq = x_sym[0], x_sym[1]
    
    # Dynamics equations
    ddq = -(g/l) * ca.sin(q) + u_sym[0]/(m * l**2)

    # State derivative (next state) = f(x,u)
    Xdot = ca.vertcat(dq, ddq)
    
    # Define the ODE system structure for CasADi
    ode = {
        'x': x_sym,  # State variables [q, dq]
        'p': u_sym,  # Parameters [tau]
        'ode': Xdot  # The Physics (The equation: dx/dt = ...)
    }
        
    # Create the numeric integrator function 'F' to integrate the dynamics
    F = ca.integrator('F',
                      'rk', # Use Runge-Kutta 4
                      ode,  # Math model defined above
                      0,    # Start Time
                      dt    # End Time ( = Step Size)
    )

    # --- Cost Function & Constraints ---
    J = 0
    Wx = ca.diag(Wx_vect) # State Weights Matrix from vector defined above
    Wu = ca.diag(Wu_vect) # Control Weights Matrix from vector defined above

    for k in range(N):
        x_k = X[:, k]
        u_k = U[:, k]
        
        # Cost accumulation
        err_x = x_k - X_targ_param
        err_u = u_k - U_targ_param
        J += err_x.T @ Wx @ err_x + err_u.T @ Wu @ err_u
        
        # Dynamics constraint 
        x_next = F(x0=x_k, p=u_k)['xf']
        opti.subject_to(X[:, k+1] == x_next)

    # Initial condition constraint
    opti.subject_to(X[:, 0] == X0_param)
    
    # --- Objective = Minimize Cost J ---
    opti.minimize(J)
    
# --- Solver Options ---
    p_opts = {
        'expand': False,   
        'print_time': False  # Don't print internal CasADi timing stats
    } 
    
    s_opts = {
        'max_iter': 500,    
        'print_level': 0,    # 0 = suppress iteration logs
        'sb': 'yes',         # Hides the initial IPOPT header info
        'tol': 1e-6          # Accuracy tolerance for convergence
    }
    
    # Solver configuration
    opti.solver('ipopt', #Use Interior Point OPTimizer 
                p_opts, 
                s_opts
    )
    
    # --- Return Values ---
    # Returns:
    # 1. opti: The NLP solver instance containing the optimization problem.
    # 2. params: Symbolic parameters for setting boundary conditions and   weights (mutable via set_value).
    # 3. variables: Symbolic expressions used to extract numerical results (trajectories, cost) after solving.
    # NOTE: Structured like this for clarity and orderliness. One could just return everything in sequence instead of this symbolic tuple.
    return opti, \
           {'X0': X0_param, 'X_targ': X_targ_param, 'U_targ': U_targ_param, 'Wx': Wx_vect, 'Wu': Wu_vect}, \
           {'J': J, 'X': X}




def generate_dataset():
    """
    Generates a dataset of optimal trajectories by solving the OCP for random initial states.
    Saves the inputs (state) and targets (optimal cost) to a file.
    
    So the file created will contain 2 arrays:
        inputs:  Nx2 array of states [q, dq]
        targets: Nx1 array of optimal costs (J)
    
    To load the dataset, use:
        data = np.load(DATA_PATH)
        X = data['inputs']   
        Y = data['targets']  
    
    Args:
        None (Parameters are defined internally: N=100, dt=0.05, Samples=1000).
    """
   
    print("--- Starting Dataset Generation... ---")
    
    N_horizon = 100  # Horizon length
    dt = 0.05        # Time step duration     
    N_samples = 1000 # Number of samples to generate (Number of OCPs solves) 
    
    opti, params, variables = create_teacher_solver(N_horizon, dt)
    
    # Set targets (Swing-up to upright)
    opti.set_value(params['X_targ'], [np.pi, 0.0]) 
    opti.set_value(params['U_targ'], [0.0])
    
    # Set weights
    opti.set_value(params['Wx'], [5.0, 0.1])
    opti.set_value(params['Wu'], [0.01])
    
    # Initialization of data set lists
    inputs = []
    targets = []
    success_count = 0

    for i in range(N_samples):
        print(f"Sample {i+1}/{N_samples}")
            
        # Set random initial state
        q_rand = np.random.uniform(0, 2*np.pi) #rad
        dq_rand = np.random.uniform(-6.0, 6.0) #rad/s
        x_init = np.array([q_rand, dq_rand])
        
        try:
            opti.set_value(params['X0'], x_init)
            sol = opti.solve() # Solve the OCP
            
            # Build dataset
            cost = sol.value(variables['J'])
            inputs.append(x_init)
            targets.append(cost)
            success_count += 1
        except Exception as e:
            if i == 0:
                print(f"[Warning] Solver failed at iter 0: {e}")
            pass 
            
    X_data = np.array(inputs)
    Y_data = np.array(targets)
    
    
    filename = 'pendulum_dataset.npz'
    
    # Save dataset to file in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'pendulum_dataset.npz')
    np.savez(filename, inputs=X_data, targets=Y_data)
    
    print(f"Dataset saved to '{filename}'. Shape: {X_data.shape}")
    print(f"Successful solves: {success_count}/{N_samples}")
    
    return opti, params, variables




def visualize_trajectory(opti, params, variables):
    """
    Solves the OCP for a specific starting state (downward position) 
    and animates the resulting swing-up trajectory.
    Just for visual validation of a OCP solution.

    Args:
        opti (casadi.Opti): The main CasADi optimization object.
        params (dict): Dictionary containing the symbolic parameters (X0, X_targ, etc.).
        variables (dict): Dictionary containing the symbolic variables (X, J).
    """
    print("\n--- Visualizing Swing-Up Trajectory ---")
    start_state = np.array([0.0, 0.0]) 
    opti.set_value(params['X0'], start_state)
    
    try:
        sol = opti.solve()
        traj_X = sol.value(variables['X']) 
        cost_val = sol.value(variables['J'])
    except Exception as e:
        print(f"Visualization failed: {e}")
        return

    # Convert to Cartesian for plotting
    l = 1.0 
    q_traj = traj_X[0, :] 
    x_bob = l * np.sin(q_traj)
    y_bob = -l * np.cos(q_traj)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(f"OCP Swing-Up (Cost: {cost_val:.2f})")
    
    line, = ax.plot([], [], 'o-', lw=2, color='blue') 
    trace, = ax.plot([], [], '-', lw=1, color='red', alpha=0.5) 
    
    def init():
        line.set_data([], [])
        trace.set_data([], [])
        return line, trace
    
    def update(frame):
        x = [0, x_bob[frame]]
        y = [0, y_bob[frame]]
        line.set_data(x, y)
        start_trace = max(0, frame-20)
        trace.set_data(x_bob[start_trace:frame], y_bob[start_trace:frame])
        return line, trace
    
    ani = animation.FuncAnimation(fig, update, frames=len(q_traj),
                                  init_func=init, interval=50, blit=True)
    plt.show()
    
    
    
    
def plot_dataset_topology(inputs, targets):
    """
    Visualizes the 'Value Function Topology' generated by the Teacher.
    It maps how the optimal cost (color) changes based on the state (position and velocity).
    
    Args:
        inputs (list or np.array): List of state vectors [angle, velocity].
        targets (list or np.array): List of corresponding optimal costs (J).
    """
    # 1. Convert to NumPy arrays for robust handling (in case lists are passed)
    X = np.array(inputs)
    Y = np.array(targets).flatten() # Flatten to 1D array for color mapping logic
    
    print(f"\n--- Plotting Value Function Landscape ({len(X)} samples) ---")

    # 2. Initialize the Figure
    plt.figure(figsize=(10, 6))
    
    # Scatter plot: x=Angle, y=Velocity, c=Cost
    sc = plt.scatter(X[:, 0], X[:, 1], c=Y, 
                     cmap='viridis', s=15, alpha=0.8, edgecolors='none')
    
    # 3. Styling and Labels
    cbar = plt.colorbar(sc)
    cbar.set_label('Optimal Cost (Value Function J)', rotation=270, labelpad=15)
    
    plt.xlabel('Angle q [rad]')
    plt.ylabel('Velocity dq [rad/s]')
    plt.title('Teacher Value Function Landscape\n(Darker regions = Lower Cost / Easier to control)')
    plt.grid(True, alpha=0.3)
    
    # Add reference lines for the origin (0,0)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

    plt.tight_layout()
    plt.show()
    



# Main Execution
if __name__ == "__main__":
    print("========================================")
    print("      PHASE 1: TEACHER EXECUTION        ")
    print("========================================")

    # STEP 1: EXAMPLE VISUAL CHECK 
    N_viz = 100
    dt_viz = 0.05
    opti_viz, params_viz, vars_viz = create_teacher_solver(N_viz, dt_viz)
    
    # Example Parameters for Visualization
    opti_viz.set_value(params_viz['X_targ'], [np.pi, 0.0]) 
    opti_viz.set_value(params_viz['U_targ'], [0.0])
    opti_viz.set_value(params_viz['Wx'], [5.0, 0.1])
    opti_viz.set_value(params_viz['Wu'], [0.01])
    
    # Example Visualization
    visualize_trajectory(opti_viz, params_viz, vars_viz)
    print("      -> Visualization check passed (or closed).")

    # STEP 2: GENERATE DATASET
    print("\n[2/3] Generating Dataset (Calling generate_dataset)...")
    
    generate_dataset() 
    
    # STEP 3: ANALYZE TOPOLOGY
    print("\n[3/3] Loading data for topology analysis...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'pendulum_dataset.npz')
    
    if os.path.exists(filename):
        data = np.load(filename)
        X_loaded = data['inputs']
        Y_loaded = data['targets']
        
        print(f"      -> Data loaded successfully from: {filename}")
        
        # 3. Visualize Topology
        plot_dataset_topology(X_loaded, Y_loaded)
    else:
        print(f"      [ERROR] File {filename} not found. Generation might have failed.")

    print("\n========================================")
    print("          PHASE 1 COMPLETE              ")
    print("========================================")