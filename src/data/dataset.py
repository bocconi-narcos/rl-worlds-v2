import torch
from torch.utils.data import Dataset

class ExperienceDataset(Dataset):
    def __init__(self, states, actions, rewards, stop_episodes, sequence_length):
        """
        states: iterable of arrays/tensors shape [1,1,H,W]
        actions: iterable of scalars or shape-[1] arrays
        rewards: iterable of scalars
        stop_episodes: iterable of booleans
        sequence_length: int, length of sequence including next state
        """
        # Stack inputs into tensors, handling list of arrays/tensors
        self.states = torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in states], dim=0)       # [N,1,1,H,W]
        self.actions = torch.stack([torch.as_tensor(a, dtype=torch.float32).view(1) for a in actions], dim=0)  # [N,1]
        self.rewards = torch.as_tensor(list(rewards), dtype=torch.float32)                                    # [N]
        self.stop_episodes = torch.as_tensor(list(stop_episodes), dtype=torch.bool)                         # [N]
        self.sequence_length = sequence_length

        # Precompute valid starting points (no episode termination within next T steps)
        self.valid_starts = []
        total = self.states.size(0)
        T = sequence_length - 1
        for i in range(total - T):
            if not self.stop_episodes[i:i + T].any():
                self.valid_starts.append(i)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        T = self.sequence_length - 1

        # State sequence: [T, 1, 1, H, W] -> [1, 1, T, H, W]
        state_seq = self.states[start:start + T].permute(1, 2, 0, 3, 4)

        # Next state: [1, 1, 1, H, W]
        next_state = self.states[start + T].unsqueeze(2)

        # Action sequence: [T, 1]
        action_seq = self.actions[start:start + T].view(T, 1)

        # Reward sequence: [T]
        reward_seq = self.rewards[start:start + T]

        print(f'State seq: {state_seq.shape}, Action seq: {action_seq.shape}, '
              f'Reward seq: {reward_seq.shape}, Next state: {next_state.shape}')

        return state_seq, action_seq, reward_seq, next_state