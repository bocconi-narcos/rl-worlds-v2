import torch
from torch.utils.data import Dataset

class ExperienceDataset(Dataset):
    def __init__(self, states, actions, rewards, stop_episodes, sequence_length=None):

        self.sequence_length = sequence_length

        self.states = states
        self.states = torch.cat(states, dim=1)
        self.actions = torch.stack(actions)
        self.rewards = torch.stack(rewards)
        self.stop_episodes = stop_episodes
        self.valid_start_indices = self.compute_valid_start_indices()

    def compute_valid_start_indices(self):
        valid_start_indices = []
        for i in range(len(self.stop_episodes) - self.sequence_length + 1):
            valid = True
            for j in range(i, i + self.sequence_length - 1):
                if self.stop_episodes[j]:
                    valid = False
                    break
            if valid:
                valid_start_indices.append(i)

        return valid_start_indices

    def __len__(self):
            # only as many as there are valid starts
            return len(self.valid_start_indices)

    def __getitem__(self, idx):
        # map `idx` → true start index
        start = self.valid_start_indices[idx]

        state = self.states[:, start:start + self.sequence_length - 1, :, :]
        next_state = self.states[:, start + self.sequence_length, :, :].unsqueeze(1)
        action = self.actions[start:start + self.sequence_length - 1]
        reward = self.rewards[start:start + self.sequence_length - 1]

        # Assert all outputs are tensors
        assert isinstance(state, torch.Tensor), "State must be a tensor"
        assert isinstance(next_state, torch.Tensor), "Next state must be a tensor"
        assert isinstance(action, torch.Tensor), "Action must be a tensor"
        assert isinstance(reward, torch.Tensor), "Reward must be a tensor"
        return state, next_state, action, reward