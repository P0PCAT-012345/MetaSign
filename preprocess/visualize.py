import matplotlib.pyplot as plt
from matplotlib import animation


def render(frames, interval=50, display=True):
    """
    Render a sequence of 2D pose frames as an animation.
    
    Args:
        frames (np.ndarray): Array of shape (frames, joints, 2)
        interval (int): Delay between frames in milliseconds
    """
    num_frames, num_joints, _ = frames.shape
    
    fig, ax = plt.subplots()
    scat = ax.scatter(frames[0, :, 0], frames[0, :, 1], s=20)  # <-- initialize with first frame!

    # Set axis limits dynamically based on your data
    all_x = frames[:, :, 0].flatten()
    all_y = frames[:, :, 1].flatten()
    margin = 0.2
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Optional, if you want (0,0) at top-left

    def update(frame_idx):
        joints = frames[frame_idx]
        scat.set_offsets(joints)
        return scat,
    
    anim = animation.FuncAnimation(
        fig, update, frames=num_frames,
        blit=False, interval=interval, repeat=False
    )
    if display:
        plt.show()
    return anim