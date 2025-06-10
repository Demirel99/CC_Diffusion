import numpy as np
import matplotlib.pyplot as plt

# --- 1. Setup and Ground Truth ---
H, W = 64, 64  # Using smaller dimensions for visualization
T = 100        # Total number of diffusion steps

# Create a sample ground truth 'x_0' with a few points
# In a real scenario, this comes from your annotations.
np.random.seed(42)
x_0 = np.zeros((H, W), dtype=np.int8)
num_people = 15
# Randomly place 'num_people' points
person_coords_y = np.random.randint(0, H, num_people)
person_coords_x = np.random.randint(0, W, num_people)
x_0[person_coords_y, person_coords_x] = 1

print(f"Initial number of points (people): {np.sum(x_0)}")
print(f"Total possible points: {H * W}")

# --- 2. Define the Noising Schedule ---
# We want to go from num_people to H*W points in T steps.
# Let's create a schedule for the *total number of ones* at each step.
# A linear schedule is a good starting point.

initial_ones = np.sum(x_0)
final_ones = H * W

# n_t = number of ones at step t
n_t_schedule = np.round(np.linspace(initial_ones, final_ones, T + 1)).astype(int)

# --- 3. Implement the Forward Process ---
def forward_noising(x_0, t, schedule):
    """
    Adds noise to x_0 to generate x_t.
    """
    # Get the target number of '1's for timestep t
    num_target_ones = schedule[t]
    
    # Get the current number of '1's
    num_current_ones = np.sum(x_0)
    
    # Calculate how many new '1's to add
    num_to_add = num_target_ones - num_current_ones
    if num_to_add <= 0:
        return x_0.copy()

    # Find the coordinates of all positions that are currently '0'
    zero_coords = np.argwhere(x_0 == 0)
    
    # Randomly select 'num_to_add' of these positions
    # Ensure we don't try to select more than are available
    num_to_add = min(num_to_add, len(zero_coords))
    indices_to_flip = np.random.choice(len(zero_coords), size=num_to_add, replace=False)
    coords_to_flip = zero_coords[indices_to_flip]
    
    # Add the new '1's
    x_t = x_0.copy()
    x_t[coords_to_flip[:, 0], coords_to_flip[:, 1]] = 1
    
    return x_t

# --- 4. Generate and Visualize the Process ---
# We can pre-generate the whole sequence for one example
# In training, you would randomly sample a 't' and generate 'x_t' on the fly.

# Let's generate the full sequence x_0, x_1, ..., x_T
xt_sequence = [x_0]
current_x = x_0
for t in range(1, T + 1):
    # This simulates the q(x_t | x_{t-1}) process iteratively
    # Note: In a real training loop, you'd compute q(x_t | x_0) directly for efficiency
    # For this discrete case, the iterative approach is simple and equivalent.
    num_ones_t_minus_1 = n_t_schedule[t-1]
    num_ones_t = n_t_schedule[t]
    num_to_add = num_ones_t - num_ones_t_minus_1

    if num_to_add > 0:
        zero_coords = np.argwhere(current_x == 0)
        indices_to_flip = np.random.choice(len(zero_coords), size=num_to_add, replace=False)
        coords_to_flip = zero_coords[indices_to_flip]
        current_x = current_x.copy()
        current_x[coords_to_flip[:, 0], coords_to_flip[:, 1]] = 1

    xt_sequence.append(current_x)


# Visualize some steps
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
time_points_to_show = [0, int(T*0.25), int(T*0.5), int(T*0.75), T-1] # Show T-1 because xt_sequence[T] is x_T

for i, t in enumerate(time_points_to_show):
    ax = axes[i]
    ax.imshow(xt_sequence[t], cmap='gray')
    num_ones = np.sum(xt_sequence[t])
    ax.set_title(f"Timestep t={t}\n{num_ones} points")
    ax.axis('off')

plt.suptitle("Forward Noising Process: Adding Points", fontsize=16)
plt.show()
