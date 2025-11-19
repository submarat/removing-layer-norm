import torch
from train import FakeLayerNorm


def test_fake_ln():
    ln = FakeLayerNorm(
        n_embd=4,
        n_ctx=2,
        layer="blocks.0.hook_resid_pre",
        bias=True,
        init_average_std=0.2,
        init_bos_std=2.0,
    )

    assert abs(ln.real_average_std_prop - 0.2) < 1e-5
    assert abs(ln.real_bos_std_prop - 2.0) < 1e-5

    assert torch.allclose(ln.real_average_std, torch.tensor(0.2, device=ln.real_average_std.device))
    assert torch.allclose(ln.real_bos_std, torch.tensor(2.0, device=ln.real_bos_std.device))

    expected_avg_std = torch.ones(4, device=ln.average_std_buffer.device) * 0.2
    expected_avg_std[0] = 2.0
    expected_bos_std = torch.ones(4, device=ln.bos_std_buffer.device) * 2.0

    assert torch.allclose(ln.average_std_buffer, expected_avg_std)
    assert torch.allclose(ln.bos_std_buffer, expected_bos_std)

    assert ln.is_fake_prop is False
    ln.is_fake_prop = True
    assert ln.is_fake_prop is True
    ln.is_fake.fill_(False)
    assert ln.is_fake_prop is False

    assert ln.bos_special_treatment_prop is True
    ln.bos_special_treatment_prop = False
    assert ln.bos_special_treatment_prop is False
    assert ln.bos_special_treatment.item() is False

    print("Basic FakeLayerNorm test passed!")


if __name__ == "__main__":
    print("Running FakeLayerNorm tests...")
    test_fake_ln()

