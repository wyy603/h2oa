import torch


class HistoryBuffer:
    def __init__(self, num_envs, num_history, num_obs, device='cuda:0'):
        """

        Parameters
        ----------
        num_envs
        num_history
        num_obs
        device

        _buffer
            shape: num_envs, num_history, num_obs
        """
        self.num_envs = num_envs
        self.num_history = num_history
        self.num_obs = num_obs
        self._buffer = torch.zeros((num_envs, num_history, num_obs),
                                   dtype=torch.float32, device=device, requires_grad=False)
        self.device = device

    def reset_idx(self, env_ids, obs=None):
        """
        fill the history buffer with the initial obs

        Parameters
        ----------
        obs

        Returns
        -------

        """
        if obs is None:
            self._buffer[env_ids] = 0.
        else:
            assert obs.shape == (len(env_ids), self.num_obs)
            self._buffer[env_ids, :] = obs.unsqueeze(1).clone()

    def update(self, obs):
        """

        Parameters
        ----------
        obs
            shape: num_envs,  num_obs

        Returns
        -------

        """
        assert obs.shape == (self.num_envs, self.num_obs)
        self._buffer = torch.cat([self._buffer[:, 1:], obs.unsqueeze(1).clone()], dim=1)

    def update_idx(self, env_ids, update_obs):
        if len(env_ids) == 0:
            return
        assert update_obs.shape == (len(env_ids), self.num_obs)
        self._buffer[env_ids] = torch.cat([self._buffer[env_ids, 1:], update_obs.unsqueeze(1)], dim=1)

    def get_history_buffer(self):
        """

        Returns
        -------
            history_buffer
                shape: (num_envs, num_history, num_obs)
                The sequence of history is (t-N, ..., t-2, t-1, t), where N is num_history
        """
        return self._buffer.clone()

    def get_flatten_history_buffer(self):
        return torch.flatten(self.get_history_buffer(), 1, 2)


# if __name__ == '__main__':
#     num_history = 5
#     num_envs = 3
#     n_obs = 4
#     buf = HistoryBuffer(num_envs, 5, n_obs)
#     buf.reset(torch.ones((num_envs, n_obs)) * 42)
#     for i in range(10):
#         env_offset = 10 * torch.arange(1, num_envs + 1, device=buf.device).view(-1, 1)
#         obs_ = torch.ones((num_envs, n_obs), device=buf.device) * i + env_offset
#         buf.update(obs_)
#         print(buf.get_history_buffer())
