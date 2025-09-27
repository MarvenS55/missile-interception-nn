import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import plotly.graph_objects as go

# --- Tool 1: Interactive Plotly Visualization  ---
def plot_interactive_plotly(interceptor_path, target_path, status):
    """
    Generates an interactive 3D plot using pplotly.
    """
    interceptor_path = np.array(interceptor_path)
    target_path = np.array(target_path)

    fig = go.Figure()

    # Add Interceptor Trajectory
    fig.add_trace(go.Scatter3d(
        x=interceptor_path[:, 0], y=interceptor_path[:, 1], z=interceptor_path[:, 2],
        mode='lines',
        line=dict(color='blue', width=5),
        name='Interceptor Path'
    ))

    # Add Target Trajectory
    fig.add_trace(go.Scatter3d(
        x=target_path[:, 0], y=target_path[:, 1], z=target_path[:, 2],
        mode='lines',
        line=dict(color='red', width=5),
        name='Target Path'
    ))

    # Add Start/End markers so you can know what happened without animations
    fig.add_trace(go.Scatter3d(x=[interceptor_path[0,0]], y=[interceptor_path[0,1]], z=[interceptor_path[0,2]], mode='markers', marker=dict(color='blue', size=8, symbol='diamond'), name='Interceptor Start'))
    fig.add_trace(go.Scatter3d(x=[target_path[0,0]], y=[target_path[0,1]], z=[target_path[0,2]], mode='markers', marker=dict(color='red', size=8, symbol='diamond'), name='Target Start'))
    if status == 'Intercept':
        # FIX: Changed 'star' to 'cross' which is a valid Plotly symbol, took me a while to figure out the correct naming conventions...
        fig.add_trace(go.Scatter3d(x=[interceptor_path[-1,0]], y=[interceptor_path[-1,1]], z=[interceptor_path[-1,2]], mode='markers', marker=dict(color='green', size=10, symbol='cross'), name='Intercept Point'))

    # Update layout for better apppearance, user can chnage this eitherway
    fig.update_layout(
        title=f'Interactive Interception Simulation<br>Result: {status}',
        scene=dict(
            xaxis_title='X Coordinate (m)',
            yaxis_title='Y Coordinate (m)',
            zaxis_title='Z Coordinate (m)',
            aspectmode='data' # This fixes the aspect ratio
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()


# --- Tool 2: Matplotlib Animation ---
def animate_matplotlib(interceptor_path, target_path, status):
    """
    Generates a 3D animation with an improved camera angle.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=25, azim=-120)

    interceptor_path = np.array(interceptor_path)
    target_path = np.array(target_path)
    max_range = np.max(np.vstack((interceptor_path, target_path)))
    min_range = np.min(np.vstack((interceptor_path, target_path)))
    ax.set_xlim([min_range, max_range]); ax.set_ylim([min_range, max_range]); ax.set_zlim([0, max_range])
    interceptor_line, = ax.plot([], [], [], lw=2, color='blue', label='Interceptor Path')
    target_line, = ax.plot([], [], [], lw=2, color='red', label='Target Path')
    interceptor_point, = ax.plot([], [], [], 'o', color='cyan', markersize=8, label='Interceptor')
    target_point, = ax.plot([], [], [], 'x', color='magenta', markersize=8, label='Target')
    ax.scatter(interceptor_path[0, 0], interceptor_path[0, 1], interceptor_path[0, 2], s=100, marker='^', label='Interceptor Start')
    ax.scatter(target_path[0, 0], target_path[0, 1], target_path[0, 2], s=100, marker='^', label='Target Start')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(f'Missile Interception Animation\nResult: {status}'); ax.legend(); ax.grid(True)

    def update(frame):
        # Update line data
        interceptor_line.set_data(interceptor_path[:frame, 0], interceptor_path[:frame, 1])
        interceptor_line.set_3d_properties(interceptor_path[:frame, 2])
        target_line.set_data(target_path[:frame, 0], target_path[:frame, 1])
        target_line.set_3d_properties(target_path[:frame, 2])

        # Update point data
        interceptor_point.set_data(interceptor_path[frame, 0:1], interceptor_path[frame, 1:2])
        interceptor_point.set_3d_properties(interceptor_path[frame, 2:3])
        target_point.set_data(target_path[frame, 0:1], target_path[frame, 1:2])
        target_point.set_3d_properties(target_path[frame, 2:3])
        return interceptor_line, target_line, interceptor_point, target_point

    ani = animation.FuncAnimation(fig, update, frames=len(interceptor_path), blit=True, interval=20)
    plt.show()

