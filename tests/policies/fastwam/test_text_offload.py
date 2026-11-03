"""Frozen text encoding can stay on CPU without changing prompt semantics."""

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from lerobot.configs import PreTrainedConfig
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.fastwam.wan.modular import FastWAM


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4))
        self.dim = 4
        self.calls = 0

    def forward(self, ids, mask):
        self.calls += 1
        return ids.unsqueeze(-1).float() * self.weight


def tokenizer(prompts, **kwargs):
    prompts = [prompts] if isinstance(prompts, str) else list(prompts)
    ids = torch.tensor([[ord(p[0]), 1, 2, 0] for p in prompts])
    return ids, ids != 0


def make_model(*, offload=None, cache=0):
    return FastWAM(
        video_expert=nn.Linear(1, 1),
        action_expert=nn.Linear(1, 1),
        mot=nn.Linear(1, 1),
        vae=nn.Linear(1, 1),
        text_encoder=TextEncoder(),
        tokenizer=tokenizer,
        device="cpu",
        text_encoder_device=offload,
        text_cache_size=cache,
    )


def test_cpu_text_stays_on_host_while_other_components_move():
    model = make_model(offload="cpu")
    model.to("meta")
    assert model.mot.weight.device.type == "meta"
    assert model.vae.weight.device.type == "meta"
    assert model.text_encoder.weight.device.type == "cpu"
    assert not model.text_encoder.weight.requires_grad
    assert not any("text_encoder" in name for name, _ in model.named_parameters())


def test_default_text_encoder_still_follows_policy_device():
    model = make_model()
    model.to("meta")
    assert model.text_encoder.weight.device.type == "meta"


def test_cached_prompts_match_uncached_outputs_without_tensor_aliases():
    model = make_model(offload="cpu", cache=2)
    reference = make_model()
    expected, expected_mask = reference.encode_prompt(["place", "press"])
    actual, mask = model.encode_prompt(["place", "press"])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(mask, expected_mask)
    actual.zero_()
    mask.zero_()
    cached, cached_mask = model.encode_prompt(["place", "press"])
    torch.testing.assert_close(cached, expected)
    torch.testing.assert_close(cached_mask, expected_mask)
    assert model.text_encoder.calls == 1
    assert torch.count_nonzero(cached[:, -1]) == 0  # preserve zeroed padding


def test_prompt_cache_is_bounded_and_preserves_prompt_order():
    model = make_model(cache=2)
    first, _ = model.encode_prompt(["place", "grasp"])
    reversed_prompts, _ = model.encode_prompt(["grasp", "place"])
    assert not torch.equal(first, reversed_prompts)
    model.encode_prompt("withdraw")
    assert len(model._text_cache) == 2
    model.encode_prompt(["place", "grasp"])
    assert model.text_encoder.calls == 4


def test_text_offload_config_round_trip_and_validation(tmp_path):
    cfg = FastWAMConfig(
        text_encoder_device="cpu", text_cache_size=8, base_model_id=None, vae_model_id="/models/wan-pinned"
    )
    cfg.save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert loaded.text_encoder_device == "cpu"
    assert loaded.text_cache_size == 8
    assert loaded.vae_model_id == "/models/wan-pinned"
    for size in (-1, True):
        with pytest.raises(ValueError, match="nonnegative integer"):
            FastWAMConfig(text_cache_size=size)
