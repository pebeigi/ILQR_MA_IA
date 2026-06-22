Extendable multi-agent ILQR vehicle experiment
==============================================

Run the default requested two-agent case:

    python3 main.py

The script writes:

    multi_agent_trajectory.csv
    multi_agent_trajectory.png
    multi_agent_trajectory_x_vs_t.png
    multi_agent_trajectory_y_vs_t.png
    multi_agent_trajectory_animation.gif

Install dependencies if needed:

    python3 -m pip install numpy matplotlib pillow

Default agents
--------------

Agent 0:
    initial state [x, y, speed, heading] = [0, 0, 2, pi/2]
    destination [x, y] = [10, 10]

Agent 1:
    initial state [x, y, speed, heading] = [5, 10, 0.1, -pi/2]
    destination [x, y] = [10, 0]

Both agents are solved over the same time grid and start at the same time.

Where to edit agents and weights
--------------------------------

Open main.py and edit the AGENTS list near the top of the file.  To add another
vehicle, append another AgentSpec entry:

    AgentSpec(
        name="agent_2",
        initial_state=np.array([x0, y0, speed0, heading0]),
        destination=np.array([xd, yd]),
        cost=CostWeights(other_agent_repulsion=25.0),
    )

No other loops need to be changed.  The dynamics, costs, CSV, and plot are built
from the length of AGENTS.

The important cost coefficients live in CostWeights:

    destination
    inverse_speed
    acceleration
    heading_rate
    static_obstacle_repulsion
    other_agent_repulsion

Dynamic inter-agent repulsion
-----------------------------

For every ordered pair of distinct agents i and j, agent i receives this running
cost while both agents are still moving:

    other_agent_repulsion / (||position_i - position_j||^2 + other_agent_epsilon)

"Still moving" is implemented as being at least STOP_RADIUS away from that
agent's own destination.  STOP_RADIUS is defined near the top of main.py.

Static obstacles
----------------

The previous stationary obstacle at [6, 5] is preserved in STATIC_OBSTACLES.  To
remove it, set:

    STATIC_OBSTACLES = []

To add more static obstacles, add more np.array([x, y]) entries to that list.

Checking sensitivity to pairwise repulsion
------------------------------------------

Run:

    python3 compare_agent_repulsion_weights.py

This writes:

    agent_repulsion_weight_sensitivity_summary.csv
    agent_repulsion_weight_sensitivity.png


Convergence outputs
-------------------
Running ``python main.py`` now also saves the ILQ convergence history:

- ``multi_agent_trajectory_convergence.csv``: one row per ILQ iteration, with each agent's total trajectory cost and the summed total cost.
- ``multi_agent_trajectory_convergence.png``: convergence curve showing cost of each agent as a function of ILQ iteration.
