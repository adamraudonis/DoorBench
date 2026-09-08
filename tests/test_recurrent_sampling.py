import numpy as np
import pytest

from doorbench.dexterous.recurrent_sampling import prefix_window, sample_windows


def test_every_admitted_label_is_reachable_without_fabricated_history():
    for length in (32,33,64,151,6457):
        covered=set()
        for label_start in range(length-32+1):
            window=prefix_window(0,label_start,episode_length=length,supervised_length=32,burn_in=64)
            assert window.observation_start==max(0,label_start-64)
            assert window.burn_in==min(label_start,64)
            assert window.label_start==label_start
            assert window.observation_start+window.total_length<=length
            covered.update(range(window.label_start,window.label_start+32))
        assert covered==set(range(length))


def test_early_labels_use_complete_available_history_and_reset_only_at_zero():
    for label in (0,1,31,32,63,64):
        window=prefix_window(0,label,episode_length=151,supervised_length=32,burn_in=64)
        assert window.observation_start==0
        assert window.burn_in==label
    assert prefix_window(0,65,episode_length=151,supervised_length=32,burn_in=64).observation_start==1


def test_legacy_random_selection_reproduces_frozen_sampler_exactly():
    old=np.random.default_rng(0);new=np.random.default_rng(0);lengths=[6457,157,151,206]
    for _ in range(5000):
        start_window=bool(old.random()<.5);burn=0 if start_window else 64;expected=[]
        for _ in range(2):
            episode=int(old.integers(4));start=0 if start_window else int(old.integers(lengths[episode]-32-burn+1))
            expected.append((episode,start,burn))
        actual=sample_windows(new,lengths,batch_size=2,supervised_length=32,burn_in=64,
            episode_start_probability=.5,mode='legacy_fixed_burn')
        assert [(w.episode,w.observation_start,w.burn_in) for w in actual]==expected


def test_corrected_frozen_seed_supervises_previously_missing_prefix():
    rng=np.random.default_rng(0);counts=np.zeros((4,64),int)
    for _ in range(1000):
        for w in sample_windows(rng,[6457,157,151,206],batch_size=2,supervised_length=32,burn_in=64,
                episode_start_probability=.5,mode='prefix_complete_v1'):
            counts[w.episode,w.label_start:min(64,w.label_start+32)]+=1
    assert (counts[:,32:64]>0).all()


def test_causal_prefix_end_cannot_be_crossed():
    with pytest.raises(ValueError,match='causal prefix'):
        prefix_window(0,120,episode_length=151,supervised_length=32,burn_in=64)
