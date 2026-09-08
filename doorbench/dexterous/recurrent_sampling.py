"""Recurrent-window indices with explicit actual-history and label coverage."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RecurrentWindow:
    episode: int
    observation_start: int
    burn_in: int
    supervised_length: int

    @property
    def label_start(self):return self.observation_start+self.burn_in

    @property
    def total_length(self):return self.burn_in+self.supervised_length


def prefix_window(episode, label_start, *, episode_length, supervised_length, burn_in):
    if (min(episode,label_start,burn_in)<0 or supervised_length<=0 or
            label_start+supervised_length>episode_length):
        raise ValueError('Supervised window must remain inside the actual causal prefix')
    history=min(label_start,burn_in)
    return RecurrentWindow(episode,label_start-history,history,supervised_length)


def sample_windows(rng, lengths, *, batch_size, supervised_length, burn_in,
                   episode_start_probability, mode='legacy_fixed_burn'):
    if mode not in ('legacy_fixed_burn','prefix_complete_v1'):
        raise ValueError('Unknown recurrent sampling contract')
    if (not lengths or min(lengths)<supervised_length+(burn_in if mode=='legacy_fixed_burn' else 0)
            or batch_size<=0 or supervised_length<=0 or burn_in<0 or
            not 0<=episode_start_probability<=1):
        raise ValueError('Invalid recurrent sampling settings')
    start_window=bool(episode_start_probability and rng.random()<episode_start_probability)
    windows=[]
    for _ in range(batch_size):
        episode=int(rng.integers(len(lengths)));n=lengths[episode]
        if mode=='legacy_fixed_burn':
            burn=0 if start_window else burn_in;total=supervised_length+burn
            start=0 if start_window else int(rng.integers(n-total+1))
            windows.append(RecurrentWindow(episode,start,burn,supervised_length))
        else:
            label_start=0 if start_window else int(rng.integers(n-supervised_length+1))
            windows.append(prefix_window(episode,label_start,episode_length=n,
                supervised_length=supervised_length,burn_in=burn_in))
    return windows
