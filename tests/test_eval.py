from types import SimpleNamespace

import numpy as np

import eval as eval_module


def test_get_dataset_uses_format_aware_loader(monkeypatch):
    loaded = object()
    call = {}

    def fake_load_dataset(name, **kwargs):
        call["name"] = name
        call["kwargs"] = kwargs
        return loaded

    monkeypatch.setattr(eval_module.swm.data, "load_dataset", fake_load_dataset)
    cfg = SimpleNamespace(
        cache_dir="/tmp/stable-wm",
        dataset=SimpleNamespace(keys_to_cache=["action", "state"]),
    )

    dataset = eval_module.get_dataset(cfg, "pusht_expert_train.lance")

    assert dataset is loaded
    assert call == {
        "name": "pusht_expert_train.lance",
        "kwargs": {
            "keys_to_cache": ["action", "state"],
            "cache_dir": "/tmp/stable-wm",
        },
    }


def test_sample_eval_starts_covers_every_valid_start_when_selecting_all():
    episodes, starts = eval_module.sample_eval_starts(
        episode_lengths=np.array([3, 5]),
        goal_offset_steps=2,
        num_eval=4,
        seed=42,
    )

    np.testing.assert_array_equal(episodes, np.array([0, 1, 1, 1]))
    np.testing.assert_array_equal(starts, np.array([0, 0, 1, 2]))
