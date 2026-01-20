import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

NUM_AGENTS = 5
SIM_TIME = 8.0
DT = 0.05
D_SAFE = 0.8
ALPHA1, ALPHA2 = 2.0, 2.0  # HO-CBF Gains
NOISE_STD = 3.0            # Every agent has their own noise

class DecentralizedAgent:
    def __init__(self, id, start_pos, goal_pos):
        self.id = id
        # State: [px, py, vx, vy]
        self.pos = np.array(start_pos, dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.goal = np.array(goal_pos, dtype=float)
        
        # Nominal Controller Parameters
        self.kp, self.kd = 2.0, 1.5

    def get_nominal_control(self):
        """
        Agents request for going to its target (Noisy PD Controller)
        """
        accel = self.kp * (self.goal - self.pos) - self.kd * self.vel
        # Local Noise (like Sensör error)
        noise = np.random.normal(0, NOISE_STD, size=accel.shape)
        
        # Physical Limitations (like Motor Limit)
        u_noisy = accel + noise
        max_accel = 5.0
        norm = np.linalg.norm(u_noisy)
        if norm > max_accel:
            u_noisy = (u_noisy / norm) * max_accel
            
        return u_noisy

    def compute_safe_control(self, neighbors):
        """
        Decentralized QP Solver
        Calculates only agent (u_i) control.
        neighbors: other agents {id: (pos, vel)} dictionary
        """
        u_nom = self.get_nominal_control()
        
        # QP variables: only its own u (2,) and slack
        u = cp.Variable(2)
        slack = cp.Variable(len(neighbors)) # For every neigbor slack
        
        constraints = []
        
        # Cntraint for every neighbor
        for k_idx, (n_id, n_state) in enumerate(neighbors.items()):
            n_pos, n_vel = n_state
            
            # Dependent States
            dp = self.pos - n_pos
            dv = self.vel - n_vel
            
            # Obstacle Values
            h = np.dot(dp, dp) - D_SAFE**2
            h_dot = 2 * np.dot(dp, dv)

            # 2*dp^T * (u_i - u_j) >= RHS
            # RHS: Push Force Required for Safety
            total_required = -2 * np.dot(dv, dv) \
                             - (ALPHA1 + ALPHA2) * h_dot \
                             - (ALPHA1 * ALPHA2) * h
            
            # Constraint: 2*dp^T * u_i >= 0.5 * Total_Required
            # Not: We assume other agent (u_j) do the other half
            Lg_Lfh_i = 2 * dp 
            
            constraints.append(
                Lg_Lfh_i @ u >= 0.5 * total_required - slack[k_idx]
            )

        # Goal: Get closer to nominal control as much as possible + punish Slack
        objective = cp.Minimize(
            cp.sum_squares(u - u_nom) + 10000 * cp.sum_squares(slack)
        )
        
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                return u.value
        except:
            pass
            
        return u_nom # If solver breaks down, implement normal

    def update(self, u_safe):
        # Double Integrator Physics: u -> v -> p
        self.vel += u_safe * DT
        self.pos += self.vel * DT


# Simulation Loop
def run_decentralized_simulation():
    # Initiate Agents 
    radius = 3.0
    angles = np.linspace(0, 2*np.pi, NUM_AGENTS, endpoint=False)
    agents = []
    
    print(f"Starting: {NUM_AGENTS} Decentralized Agent.")
    
    for i in range(NUM_AGENTS):
        start = [radius * np.cos(angles[i]), radius * np.sin(angles[i])]
        goal = [-start[0], -start[1]]
        agents.append(DecentralizedAgent(i, start, goal))

    trajectory = np.zeros((int(SIM_TIME/DT), NUM_AGENTS, 2))
    
    for t in range(int(SIM_TIME/DT)):
        # Every agent senses or understands the environment
        current_states = {a.id: (a.pos, a.vel) for a in agents}
        
        actions = []
        
        # Every Agent gives their own decision
        for agent in agents:
            # Filter Neighbors
            neighbors = {nid: s for nid, s in current_states.items() if nid != agent.id}
            
            # Decentralized Calculation
            u_safe = agent.compute_safe_control(neighbors)
            actions.append(u_safe)
            
        # Adım 3: Update Physics
        for i, agent in enumerate(agents):
            agent.update(actions[i])
            trajectory[t, i, :] = agent.pos
            
        if t % 20 == 0: print(f"Step {t}")
            
    return trajectory, agents

# GIF
def save_gif(trajectory, agents):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.grid(True, alpha=0.3)
    ax.set_title("Decentralized Multi-Agent Safety (Distributed CBF)")
    
    colors = plt.cm.spring(np.linspace(0, 1, NUM_AGENTS))
    circles = [plt.Circle((0,0), 0.4, color=c, alpha=0.8) for c in colors]
    trails = [ax.plot([], [], color=c, alpha=0.5)[0] for c in colors]
    
    for i, a in enumerate(agents):
        ax.scatter(a.goal[0], a.goal[1], color=colors[i], marker='x')
        ax.add_patch(circles[i])

    def update(frame):
        for i, circle in enumerate(circles):
            pos = trajectory[frame, i]
            circle.center = pos
            # İz
            hist = trajectory[max(0, frame-30):frame+1, i]
            trails[i].set_data(hist[:,0], hist[:,1])
        return circles + trails

    anim = FuncAnimation(fig, update, frames=len(trajectory), blit=True)
    anim.save('decentralized_swarm.gif', writer='pillow', fps=30)
    print("GIF is ready")

if __name__ == "__main__":
    traj, agent_objs = run_decentralized_simulation()
    save_gif(traj, agent_objs)