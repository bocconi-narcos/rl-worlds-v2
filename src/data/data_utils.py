import os
import random
import multiprocessing
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from data.dataset import ExperienceDataset
from stable_baselines3.common.vec_env import SubprocVecEnv
from data.rl_agent import create_ppo_agent, train_ppo_agent, save_ppo_agent
from data.env_utils import ActionRepeatWrapper, ImagePreprocessingWrapper, FrameStackWrapper

def _load_existing_dataset(config: dict):
    """Attempts to load a pre-existing dataset based on the configuration."""
    load_path = config['data_collection'].get('load_path')
    dataset_dir = 'datasets/'
    env_name = config['environment']['name']

    if not load_path:
        print("`data.dataset.load_path` is not specified. Proceeding to collect new data.")
        return None

    dataset_path = os.path.join(dataset_dir, load_path)
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset not found at {dataset_path}. Proceeding to collect new data.")
        return None

    print(f"Loading dataset from {dataset_path}...")
    try:
        data = torch.load(dataset_path, weights_only=False)
        metadata = data.get('metadata', {})
        
        # --- Validate Metadata ---
        if metadata.get('environment_name') != env_name:
            raise ValueError(f"Environment mismatch: config is '{env_name}', but dataset is for '{metadata.get('environment_name')}'.")
        
        if metadata.get('collection_method') != 'ppo':
            print(f"Warning: Loaded dataset was not collected via PPO (method: {metadata.get('collection_method', 'unknown')}). Collecting new data.")
            return None

        print(f"Successfully loaded PPO-collected dataset for '{env_name}'.")
        return data['train_dataset'], data['val_dataset']

    except Exception as e:
        print(f"Error loading or validating dataset from {dataset_path}: {e}. Proceeding to collect new data.")
        return None


def _initialize_environment(config: dict):
    """Creates and wraps a single environment instance for data collection."""

    # Get environment name from config
    env_name = config['environment']['name']

    # if env_name starts with 'ALE-', we need to strip 
    if env_name.startswith('ALE'):
        import ale_py
        gym.register_envs(ale_py)
    print(f"Creating environment: {env_name}")
    # Determine the correct render mode
    render_mode = 'rgb_array'
    try:
        env = gym.make(env_name, render_mode=render_mode)
    except Exception as e:
        print(f"Could not create env with render_mode='rgb_array' ({e}). Trying render_mode=None.")
        render_mode = None
        env = gym.make(env_name, render_mode=render_mode)

    # Apply preprocessing wrappers
    img_h = config['environment'].get('image_height')
    img_w = config['environment'].get('image_width')
    img_size = (img_h, img_w)
    env = ImagePreprocessingWrapper(env, img_size, grayscale=True)

    # Apply action repetition wrapper if configured
    ppo_config = config.get('ppo_agent', {})
    action_repetition_k = ppo_config.get('action_repetition_k', 1)
    if action_repetition_k > 1:
        print(f"Applying ActionRepeatWrapper with k={action_repetition_k}.")
        env = ActionRepeatWrapper(env, action_repetition_k)

    frame_stack_size = config['environment'].get('frame_stack_size')
    env = FrameStackWrapper(env, stack_size=frame_stack_size)  # Stack 4 frames for temporal context
    
    return env, render_mode


def _train_agent(config: dict) -> PPO:
    """Sets up a vectorized environment and trains a PPO agent."""

    print("\n--- Setting up and Training PPO Agent ---")
    
    ppo_config = config.get('data_collection').get('ppo_agent')
    load_agent = config['data_collection'].get('ppo_agent', {}).get('load')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device  # Use MPS if available
    
    # Create vectorized environment for training
    n_envs = ppo_config.get('n_envs', multiprocessing.cpu_count())

    
    env_fns = [
    (lambda cfg=config: _initialize_environment(cfg)[0])
    for _ in range(n_envs)
    ]
        
    vec_env = SubprocVecEnv(env_fns)
    
    # Create and train the agent
    agent = create_ppo_agent(vec_env, ppo_config, device=device, load_agent=load_agent)
    train_ppo_agent(agent, ppo_config, task_name="PPO Training for Data Collection")
    
    # Optionally add noise to the policy for exploration during collection
    noise = ppo_config.get('additional_log_std_noise', 0.0)
    if noise != 0.0 and hasattr(agent.policy, 'log_std'):
        agent.policy.log_std.data += torch.tensor(noise, device=device)
        print(f"Adjusted PPO policy log_std by {noise:.4f} for collection.")

    # Save the trained agent
    save_path = 'best_models/ppo_agent'
    save_ppo_agent(agent, save_path)

    vec_env.close() # Close the training env; we'll use a single env for collection
    return agent


def _collect_episodes_with_agent(env: gym.Env, agent: PPO, config: dict):
    """Collects trajectories from the environment using the trained agent."""
    print("\n--- Collecting Episodes with Trained PPO Agent ---")
    
    # Get image size from config
    num_episodes = config['data_collection']['num_episodes']
    max_steps = config['data_collection']['max_steps_per_episode']

    # Get random action percentage from config, default 0.0
    random_action_pct = config['data_collection'].get('random_action_percentage')
    all_episodes_data = []

    for i in range(num_episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        step_count = 0
        total_reward = 0
        episode_transitions = []

        while not (terminated or truncated) and step_count < max_steps:
            # Decide whether to use random action or agent
            if random.random() < random_action_pct:
                action = env.action_space.sample()
            else:
                action, _ = agent.predict(obs, deterministic=False)

            # Perform step
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Store transition (obs are already preprocessed numpy arrays)
            transition = (
                torch.from_numpy(obs).float(),
                action,
                reward,
                torch.from_numpy(next_obs).float(),
                terminated or truncated
            )
            episode_transitions.append(transition)

            obs = next_obs
            total_reward += reward
            step_count += 1

        all_episodes_data.append(episode_transitions)
        print(f"Episode {i+1}/{num_episodes} finished. Steps: {step_count}, Reward: {total_reward:.2f}")
    return all_episodes_data


def _create_and_split_datasets(episodes_data: list, config: dict):
    """Shuffles, splits, and converts raw episode data into ExperienceDataset objects."""
    if not episodes_data:
        print("Warning: No data was collected. Returning empty datasets.")
        sequence_length = config['data_collection'].get('sequence_length')
        return ExperienceDataset([], [], [], [], sequence_length=sequence_length), ExperienceDataset([], [], [], [], sequence_length=sequence_length)

    random.shuffle(episodes_data)
    
    split_ratio = config['data_collection']['validation_split']
    split_idx = int(len(episodes_data) * (1.0 - split_ratio))
    
    train_episodes = episodes_data[:split_idx]
    val_episodes = episodes_data[split_idx:]
    
    print(f"\nSplitting {len(episodes_data)} collected episodes into {len(train_episodes)} train and {len(val_episodes)} validation.")

    def flatten_episodes(episodes):
        states = []
        actions = []
        rewards = []
        stop_episodes = []

        for episode in episodes:
            for idx, (s, a, r, ns, se) in enumerate(episode):
                # if it is the last transition of the episode, make se True


                s = s.unsqueeze(0)  # Add channel dimension
                state = s[:, -1, :, :]
                a = torch.tensor(a, dtype=torch.float32)
                r = torch.tensor(r, dtype=torch.float32)
                states.append(state.unsqueeze(0))
                actions.append(a)
                rewards.append(r)
                if idx == len(episode) - 1:
                    se = True
                stop_episodes.append(se)

                if se:
                    actions.append(a)  # Append action for the last transition
                    rewards.append(r)  # Append reward for the last transition
                    ns.unsqueeze_(0)  # Add channel dimension for next state
                    next_state = ns[:, -1, :, :]
                    states.append(next_state.unsqueeze(0))

                #assert False

        return states, actions, rewards, stop_episodes

    train_s, train_a, train_r, train_se = flatten_episodes(train_episodes)
    val_s, val_a, val_r, val_se = flatten_episodes(val_episodes)

    sequence_length = config['data_collection'].get('sequence_length')
    
    # Data is already preprocessed, so transform is None
    train_dataset = ExperienceDataset(train_s, train_a, train_r, train_se, sequence_length=sequence_length)
    val_dataset = ExperienceDataset(val_s, val_a, val_r, val_se, sequence_length=sequence_length)

    print(f"Created training dataset with {len(train_dataset)} transitions.")
    print(f"Created validation dataset with {len(val_dataset)} transitions.")
    
    return train_dataset, val_dataset


def _save_new_dataset(train_dataset, val_dataset, config: dict):
    """Saves the newly collected datasets along with metadata."""

    max_steps = config['data_collection']['max_steps_per_episode']
    img_h = config['environment'].get('image_height')
    img_w = config['environment'].get('image_width')
    
    # Create datasets directory if it doesn't exist
    dataset_dir = 'datasets'
    save_filename = config['data_collection']['filename']
    os.makedirs(dataset_dir, exist_ok=True)
    save_path = os.path.join(dataset_dir, save_filename)
    
    ppo_config = config.get('data_collection', {}).get('ppo_agent', {})
    
    metadata = {
        'environment_name': config['environment']['name'],
        'collection_method': 'ppo',
        'num_episodes_collected': config['data_collection']['num_episodes'],
        'image_height': img_h,
        'image_width': img_w,
        'max_steps_per_episode': max_steps,
        'validation_split_ratio': config['data_collection']['validation_split'],
        'frame_skipping': config['environment'].get('frame_skipping', 'N/A'),
        'action_repetition_k': ppo_config.get('action_repetition_k', 1),
        'num_train_transitions': len(train_dataset),
        'num_val_transitions': len(val_dataset),
        'ppo_config_params': ppo_config,
    }
    
    data_to_save = {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'metadata': metadata
    }
    
    try:
        torch.save(data_to_save, save_path)
        print(f"\nSuccessfully saved new dataset to {save_path}")
    except Exception as e:
        print(f"Error: Failed to save dataset to {save_path}: {e}")


# --- Main Orchestrator Function ---

def collect_ppo_episodes(config):
    """
    Loads or collects robotics experience data using a PPO agent.

    This function orchestrates the process:
    1. Tries to load a pre-existing, validated dataset.
    2. If unavailable, it initializes and trains a PPO agent.
    3. Uses the trained agent to collect new episode data.
    4. Splits the data into training and validation sets.
    5. Saves the new dataset for future use.
    
    Args:
        config (dict): A dictionary containing all configuration parameters.

    Returns:
        tuple: A tuple containing the training ExperienceDataset and validation ExperienceDataset.
    """
    # 1. Attempt to load an existing dataset
    loaded_data = _load_existing_dataset(config)
    if loaded_data:
        return loaded_data

    # --- If no data was loaded, proceed with collection ---
    
    # 2. Initialize a single, wrapped environment for collection
    # We create this first to determine the correct render_mode for the training envs
    collection_env, render_mode = _initialize_environment(config)
    
    # 3. Train the PPO agent
    agent = _train_agent(config)

    # 4. Collect episodes using the trained agent and the single environment
    raw_episodes = _collect_episodes_with_agent(collection_env, agent, config)
    collection_env.close()

    # 5. Create and split datasets from the collected raw data
    train_dataset, val_dataset = _create_and_split_datasets(raw_episodes, config)

    print('len(train_dataset) in ppo: ', len(train_dataset))
    print('len(val_dataset) in ppo: ', len(val_dataset) if val_dataset else 0)

    # 6. Save the new datasets if any data was collected
    if len(train_dataset) > 0 or len(val_dataset) > 0:
        _save_new_dataset(train_dataset, val_dataset, config)
    
    return train_dataset, val_dataset