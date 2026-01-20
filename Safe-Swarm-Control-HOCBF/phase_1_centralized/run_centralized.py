import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Configuration & Parameters
NUM_AGENTS = 5
SIM_TIME = 8.0     # Total simulation time (s)
DT = 0.05      # Time step (s)
D_SAFE = 0.8          # Minimum safety distance
ALPHA1 = 2.0   # HO-CBF Gain 1 (Position error decay)
ALPHA2 = 2.0        # HO-CBF Gain 2 (Velocity error decay)
NOISE_STD = 3.0  # High noise to simulate a VERY bad pilot
SLACK_PENALTY = 10000    # Penalty for violating safety

# Physics & Nominal Policy
def nominal_controller(positions, velocities, goals):
    """
    A 'Dumb' PD Controller with heavy Gaussian Noise.
    This represents an unstable Learning Policy.
    """
    kp, kd = 2.0, 1.5
    # PD Control: u = Kp*error - Kd*velocity
    u_nominal = kp * (goals - positions) - kd * velocities
    
    # Chaos (Gaussian Noise)
    noise = np.random.normal(0, NOISE_STD, size=u_nominal.shape)
    u_noisy = u_nominal + noise
    
    # Clip max acceleration for physical realism (motor limits)
    max_accel = 5.0
    norms = np.linalg.norm(u_noisy, axis=1, keepdims=True)
    factor = np.clip(max_accel / (norms + 1e-6), 0, 1)
    return u_noisy * factor

# High-Order CBF (Degree 2)
def get_ho_cbf_constraints(positions, velocities, u_nom):
    """
    Constructs the A*u <= b matrices for the Double Integrator CBF.
    """
    N = len(positions)
    
    # Optimization Variables
    u = cp.Variable((N, 2))      # Control Input
    slack = cp.Variable(N * (N - 1) // 2) # Slack to prevent infeasibility
    
    constraints = []
    k = 0 # Index for slack variable
    
    for i in range(N):
        for j in range(i + 1, N):
            # State differences
            dp = positions[i] - positions[j]
            dv = velocities[i] - velocities[j]
            
            # Barrier Function: h = ||dp||^2 - D^2
            dist_sq = np.dot(dp, dp)
            h = dist_sq - D_SAFE**2
            
            # Derivative: h_dot = 2 * dp^T * dv
            h_dot = 2 * np.dot(dp, dv)
            
            # d/dt(h_dot + alpha1*h) + alpha2*(h_dot + alpha1*h) >= 0
            # Becomes: 2*dp^T * (ui - uj) >= ...
            
            Lg_Lfh = 2 * dp # Coefficient for u
            Lf_Lfh = 2 * np.dot(dv, dv) # Drift term
            
            # The Constraint: Lg_Lfh * u >= -Lf_Lfh - terms
            # We add slack[k] to the right side to allow slight violation if trapped
            lhs = Lg_Lfh @ (u[i] - u[j])
            rhs = -Lf_Lfh - (ALPHA1 + ALPHA2) * h_dot - (ALPHA1 * ALPHA2) * h
            
            constraints.append(lhs >= rhs - slack[k])
            k += 1
            
    return u, slack, constraints

# The Solver Loop
def run_simulation():
    # Initialize Agents in a Circle
    angles = np.linspace(0, 2*np.pi, NUM_AGENTS, endpoint=False)
    radius = 3.0
    
    # State: [Position, Velocity]
    positions = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    velocities = np.zeros_like(positions)
    
    # Goal: opposite side
    goals = -positions.copy()
    
    trajectory = [positions.copy()]
    
    print(f"Starting Double Integrator Simulation ({NUM_AGENTS} Agents).")
    
    steps = int(SIM_TIME / DT)
    for t in range(steps):
        # 1. Nominal (Unsafe) Control
        u_nom = nominal_controller(positions, velocities, goals)
        
        # QP Constraints
        u_var, slack_var, constraints = get_ho_cbf_constraints(positions, velocities, u_nom)
        
        # Objective: Minimize deviation from nominal + Penalize Slack
        objective = cp.Minimize(
            cp.sum_squares(u_var - u_nom) + SLACK_PENALTY * cp.sum_squares(slack_var)
        )
        
        # 4. Solve QP
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            
            if prob.status in ["optimal", "optimal_inaccurate"]:
                u_safe = u_var.value
            else:
                print(f"Solver failed at step {t}, using nominal.")
                u_safe = u_nom
        except cp.SolverError:
            u_safe = u_nom

        # 5. Physics Update (Double Integrator: u -> v -> p)
        velocities += u_safe * DT
        positions += velocities * DT
        
        trajectory.append(positions.copy())
        
        if t % 20 == 0:
            print(f"Step {t}/{steps} complete")

    return np.array(trajectory), goals

# GIF Generation
def save_gif(trajectory, goals, filename="centralized_double_int.gif"):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Setup Plot
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Safety Filter\nDouble Integrator Dynamics ($\ddot{{x}}=u$)")
    
    # Draw Goals
    ax.scatter(goals[:,0], goals[:,1], c='green', marker='x', s=100, label="Goals")
    
    # Create Agent Circles
    colors = plt.cm.plasma(np.linspace(0, 1, NUM_AGENTS))
    agents = [plt.Circle((0,0), D_SAFE/2, color=colors[i], alpha=0.9) for i in range(NUM_AGENTS)]
    trails = [ax.plot([], [], color=colors[i], alpha=0.4)[0] for i in range(NUM_AGENTS)]
    
    for agent in agents: ax.add_patch(agent)

    def init():
        return agents + trails

    def update(frame):
        current_pos = trajectory[frame]
        for i, agent in enumerate(agents):
            agent.center = current_pos[i]
            # Draw Trail
            hist = trajectory[max(0, frame-20):frame+1, i, :]
            trails[i].set_data(hist[:, 0], hist[:, 1])
        return agents + trails

    anim = FuncAnimation(fig, update, frames=len(trajectory), init_func=init, blit=True)
    anim.save(filename, writer='pillow', fps=30)
    print(f"Animation saved to {filename}")

if __name__ == "__main__":
    traj, final_goals = run_simulation()
    save_gif(traj, final_goals)