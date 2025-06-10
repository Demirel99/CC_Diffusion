# Point-Diffusion: A Discrete Diffusion Framework for Point-Based Object Localization

This repository outlines a novel framework, **Point-Diffusion**, for point-based object localization and counting. It reframes the task from a traditional regression or density-map estimation problem into a **generative denoising process on a discrete, binary point map**.

Our core idea is to start with a map of random "false positive" points and train a model to iteratively remove them, conditioned on an input image, until only the true object locations remain. This approach simplifies the localization pipeline and leverages the power of diffusion models in a domain-specific way.

---

## 💡 Core Idea

Traditional methods for object localization often regress coordinates or estimate continuous density maps. These can be complex, require careful tuning of loss functions, and often rely on non-differentiable post-processing steps like non-maximum suppression or finding local maxima.

**Point-Diffusion proposes a simpler, more intuitive paradigm:**

1.  **The "Noise" is False Positives:** Instead of adding Gaussian noise to pixel values, our "noise" is the **addition of spurious points** to a coordinate map. The noisiest possible state (`x_T`) is a map where every single pixel is a potential object. The clean state (`x_0`) is a sparse map with points only at the ground truth locations.

2.  **The "Denoising" is Removing False Positives:** The model's task is to act as a "denoiser." Given a noisy map with many points and the context from the real image, it learns to predict which points are noise (false positives) and should be removed.

### Visualizing the Process

The entire process can be visualized as moving between a clean ground-truth map (`x_0`) and a fully noisy map (`x_T`).

**Forward Process (Adding Noise):** We define a schedule to progressively add random points to the ground truth map until it's completely filled.


*A simple visualization of the forward process. `x_0` is the sparse ground truth. `x_t` is a partially noised version. `x_T` is the fully noised state.*

**Reverse Process (Model Denoising):** The model learns the reverse. It starts with a fully random map (`x_T`) and, guided by the image, iteratively removes points to arrive at a clean prediction of the true object locations.

---

## ✨ Key Advantages

*   **Conceptual Simplicity:** Eliminates the need for complex anchor generation, bounding box regression, or density map conversion. The model's task is a simple, intuitive binary classification at each point: "Is this point a real object or not?"

*   **Domain-Specific Noise Model:** The corruption process (adding false positives) directly mirrors the core challenge of the detection task (discriminating true from false positives).

*   **End-to-End Training:** The framework is fully differentiable. The final output is a probability map that can be directly used for localization and counting (`count = sum(probabilities > threshold)`).

*   **Robustness and Generative Power:** The iterative nature of the denoising process allows the model to refine its predictions, potentially handling occluded and ambiguous scenes better than single-pass regression models.

---

## ⚙️ Methodology

Our framework consists of a **Forward Noising Process** (fixed) and a **Reverse Denoising Process** (learned).

### 1. The Forward Process (`q`)

The forward process, `q(x_t | x_0)`, defines how to generate a "noisy" point map `x_t` from a clean ground truth map `x_0`.

*   **State Representation:** We use a binary map `x` of shape `(H, W)`, where `x[i, j] = 1` if a point exists at `(i, j)` and `0` otherwise. `x_0` is the ground truth map containing `N` points.

*   **Noise Schedule:** We define a schedule that determines the total number of points, `n_t`, that should exist in the map at timestep `t`. A simple linear schedule is effective:
    `n_t = round(N + (t/T) * (H*W - N))`
    where `T` is the total number of timesteps.

*   **Transition `q(x_t | x_0)`:** To generate `x_t` directly from `x_0` for efficient training:
    1.  Calculate the number of noise points to add: `k = n_t - N`.
    2.  Identify all coordinates where `x_0` is `0`.
    3.  Randomly sample `k` of these "zero" coordinates.
    4.  Flip their values to `1`. The result is the noisy map `x_t`.

### 2. The Reverse Process (`p_θ`)

The reverse process learns to denoise the map using a neural network `p_θ`, which is conditioned on the image and the current timestep.

*   **Model Goal:** The model `p_θ(x_0 | x_t, image, t)` is trained to predict the **original clean map `x_0`** given a noisy map `x_t`, the corresponding image, and the timestep `t`.

*   **Model Architecture:** A **U-Net** is an excellent choice due to its strong performance in image-to-image translation tasks.
    *   **Input:** The model takes the input `image`, the noisy map `x_t`, and a sinusoidal time embedding of `t`, often concatenated on the channel axis.
    *   **Output:** A logit map of shape `(H, W)` where each value represents the probability of a point existing at that location.

*   **Loss Function:** Since the task is to predict a binary map (`x_0`) from a probability map (model output), the **Binary Cross-Entropy (BCE) Loss** is a natural and effective choice.
    `Loss = BCE(model_output, x_0)`

---

## 🚀 Getting Started: A Training Loop Example

Here is a high-level pseudo-code implementation of the training logic.

```python
# Model, optimizer, and schedule are pre-defined
model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
T = 1000 # Total timesteps
n_t_schedule = ... # Pre-calculate the number of points at each step t

def forward_noising(x_0, t, schedule):
    # Implements the direct q(x_t | x_0) sampling described above
    num_target_ones = schedule[t]
    num_gt_ones = x_0.sum()
    num_to_add = num_target_ones - num_gt_ones

    if num_to_add <= 0:
        return x_0.clone()

    zero_coords = (x_0 == 0).nonzero()
    indices_to_flip = torch.randperm(len(zero_coords))[:num_to_add]
    coords_to_flip = zero_coords[indices_to_flip]

    x_t = x_0.clone()
    x_t[coords_to_flip[:, 0], coords_to_flip[:, 1]] = 1
    return x_t


# --- Training Loop ---
for image, ground_truth_points in dataloader:
    optimizer.zero_grad()

    # 1. Create the ground truth binary map x_0
    x_0 = create_binary_map(ground_truth_points, H, W)

    # 2. Sample a random timestep t for each item in the batch
    t = torch.randint(1, T + 1, (batch_size,))

    # 3. Generate the noisy map x_t using the forward process
    x_t = forward_noising(x_0, t, n_t_schedule)

    # 4. Get the model's prediction of the clean map
    predicted_x_0 = model(image, x_t, t) # t is passed as a time embedding

    # 5. Calculate loss
    loss = binary_cross_entropy(predicted_x_0, x_0)

    # 6. Backpropagate and update
    loss.backward()
    optimizer.step()
