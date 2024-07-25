import numpy as np
import matplotlib.pyplot as plt


def generate_measurements(target_positions, num_measurements_per_target, noise_cov):
    measurements = []
    for pos in target_positions:
        for _ in range(num_measurements_per_target):
            noise = np.random.multivariate_normal([0, 0], noise_cov)
            measurements.append(pos + noise)
    return np.array(measurements)


def jdpa_update(target_positions, measurements, noise_cov):
    num_targets = len(target_positions)
    num_measurements = len(measurements)
    association_probs = np.zeros((num_targets, num_measurements))

    for i, pos in enumerate(target_positions):
        for j, z in enumerate(measurements):
            diff = z - pos
            exponent = -0.5 * np.dot(diff.T, np.linalg.inv(noise_cov).dot(diff))
            association_probs[i, j] = np.exp(exponent)

    association_probs /= association_probs.sum(axis=1, keepdims=True)  # Normalize probabilities

    updated_positions = []
    for i in range(num_targets):
        new_pos = np.sum(association_probs[i][:, np.newaxis] * measurements, axis=0)
        updated_positions.append(new_pos)

    return np.array(updated_positions)


# Initialize target positions and noise covariance
target_positions = [np.array([2.0, 3.0]), np.array([5.0, 7.0])]
noise_cov = np.array([[0.5, 0], [0, 0.5]])
num_measurements_per_target = 5

# Generate measurements
measurements = generate_measurements(target_positions, num_measurements_per_target, noise_cov)

# Update target positions using JDPA
updated_positions = jdpa_update(target_positions, measurements, noise_cov)

# Plotting
plt.scatter(measurements[:, 0], measurements[:, 1], c='r', label='Measurements')
plt.scatter([pos[0] for pos in target_positions], [pos[1] for pos in target_positions], c='g', label='True Positions')
plt.scatter([pos[0] for pos in updated_positions], [pos[1] for pos in updated_positions],
            c='b',
            label='Updated Positions (JDPA)')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('JDPA Algorithm Simulation')
plt.show()
