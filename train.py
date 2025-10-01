import tensorflow as tf
import h5py
import numpy as np
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Looad thhe dataset
with h5py.File('data/dataset.h5', 'r') as hf:
    states = hf['states'][:]
    actions = hf['actions'][:]
    rewards = hf['rewards'][:]
    dones = hf['dones'][:]

# 2. Filter to include only successful runs
episode_ends = np.where(dones)[0]
episode_starts = np.roll(episode_ends + 1, 1)
episode_starts[0] = 0

success_indices = []
for start, end in zip(episode_starts, episode_ends):
    # An episode is successful if its its final reward is high (our +100 bonus)
    if rewards[end] >= 50:
        success_indices.extend(range(start, end + 1))

states_filtered = states[success_indices]
actions_filtered = actions[success_indices]
print(f"Filtered dataset from {len(states)} to {len(states_filtered)} samples from successful episodes.")

# 3. normalize the data
state_scaler = StandardScaler().fit(states_filtered)
action_scaler = StandardScaler().fit(actions_filtered)

states_norm = state_scaler.transform(states_filtered)
actions_norm = action_scaler.transform(actions_filtered)

# 4. Split data for training and vvalidation
states_train, states_val, actions_train, actions_val = train_test_split(
    states_norm, actions_norm, test_size=0.2, random_state=42
)

# 5. define and compile the model
model = tf.keras.Sequential([
    layers.Dense(128, input_shape=(6,), activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(3)  # Output is the 3D thrust vector
])
model.compile(optimizer='adam', loss='mse')

# 6. train the moodel i found 50 epochs to be the sweet spot
history = model.fit(
    states_train,
    actions_train,
    epochs=50,
    batch_size=64,
    validation_data=(states_val, actions_val),
    verbose=2
)

# 7. Save the trained model 
model.save('models/dqn_model.keras')
joblib.dump(state_scaler, 'models/state_scaler.pkl')
joblib.dump(action_scaler, 'models/action_scaler.pkl')

print("\nModel and scalers saved successfully.")
print(f"Final validation loss: {history.history['val_loss'][-1]}")