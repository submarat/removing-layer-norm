import torch
from train import FakeLayerNorm
from std_dicts import std_dicts

def test_fake_ln():
    ln = FakeLayerNorm(n_embd=4, n_ctx=2, layer="blocks.0.hook_resid_pre", bias=True, init_average_std=0.2, init_bos_std=2.0)

    # Test initialization
    assert ln.real_average_std == 0.2
    assert ln.real_bos_std == 2.0
    assert torch.allclose(ln.average_std.cpu(), torch.tensor([2.0, 0.2, 0.2, 0.2]))
    assert torch.allclose(ln.bos_std.cpu(), torch.tensor([2.0, 2.0, 2.0, 2.0]))


def test_serialization():
    import os
    import torch
    from transformers import GPT2Config, AutoTokenizer
    from train import load_model
    import tempfile
    import shutil
    from safetensors.torch import load_file

    print("Starting FakeLayerNorm serialization test...")
    
    # Create a temporary directory for saving the model
    with tempfile.TemporaryDirectory() as temp_dir:
        # Load GPT-2 with FakeLayerNorm
        print("Loading GPT-2 model with FakeLayerNorm...")
        model = load_model(model_name="gpt2", remove_ln=True)
        print("Model loaded successfully.")

        # Modify some attributes to verify they're preserved
        block0_ln1 = model.transformer.h[0].ln_1
        block0_ln2 = model.transformer.h[0].ln_2
        final_ln = model.transformer.ln_f

        print("Setting custom values for FakeLayerNorm attributes...")
        # Modify attributes
        block0_ln1.is_fake = True
        block0_ln1.attn_v_is_fake = True
        block0_ln1.real_average_std = 0.5
        block0_ln1.real_bos_std = 2.5
        
        # Change some tensor values
        with torch.no_grad():
            block0_ln1.average_std = torch.ones_like(block0_ln1.average_std) * 0.6
            block0_ln1.average_std[0] = 2.6
            block0_ln1.bos_std = torch.ones_like(block0_ln1.bos_std) * 2.7

        # Also modify ln_2
        block0_ln2.is_fake = False
        block0_ln2.real_average_std = 0.7
        
        # And final ln
        final_ln.is_fake = True
        final_ln.real_bos_std = 3.0

        # Print initial values
        print(f"Original ln_1 values: is_fake={block0_ln1.is_fake}, attn_v_is_fake={block0_ln1.attn_v_is_fake}")
        print(f"Original ln_1 std values: real_avg={block0_ln1.real_average_std}, real_bos={block0_ln1.real_bos_std}")
        print(f"Original ln_1 average_std[0]={block0_ln1.average_std[0].item()}, average_std[1]={block0_ln1.average_std[1].item()}")
        print(f"Original ln_1 bos_std[0]={block0_ln1.bos_std[0].item()}, bos_std[1]={block0_ln1.bos_std[1].item()}")
        print(f"Original ln_2 values: is_fake={block0_ln2.is_fake}, real_average_std={block0_ln2.real_average_std}")
        
        # Save the model using standard PyTorch save
        print(f"Saving model to {temp_dir}/test_model...")
        save_path = os.path.join(temp_dir, "test_model")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path, safe_serialization=False)  # Use pytorch_model.bin format
        print("Model saved successfully.")
        
        # List files in the directory to check what's saved
        print("Files in the saved model directory:")
        for file in os.listdir(save_path):
            print(f" - {file}")

        # Now load the model back
        print("Loading model back from saved state...")
        loaded_model = load_model(model_name="gpt2", remove_ln=True)
        
        # Determine which format the model was saved in
        if os.path.exists(os.path.join(save_path, "pytorch_model.bin")):
            state_dict = torch.load(os.path.join(save_path, "pytorch_model.bin"))
        else:
            state_dict = load_file(os.path.join(save_path, "model.safetensors"))
        
        # Debug: Print keys in state dict related to ln_2
        print("\nDebug - State dict keys related to ln_2:")
        for key in [k for k in state_dict.keys() if "ln_2" in k]:
            print(f"  {key}")
        print("")
            
        loaded_model.load_state_dict(state_dict)
        print("Model loaded successfully.")

        # Get the loaded versions of the components
        loaded_block0_ln1 = loaded_model.transformer.h[0].ln_1
        loaded_block0_ln2 = loaded_model.transformer.h[0].ln_2
        loaded_final_ln = loaded_model.transformer.ln_f
        
        # Print loaded values
        print(f"Loaded ln_1 values: is_fake={loaded_block0_ln1.is_fake}, attn_v_is_fake={loaded_block0_ln1.attn_v_is_fake}")
        print(f"Loaded ln_1 std values: real_avg={loaded_block0_ln1.real_average_std}, real_bos={loaded_block0_ln1.real_bos_std}")
        print(f"Loaded ln_1 average_std[0]={loaded_block0_ln1.average_std[0].item()}, average_std[1]={loaded_block0_ln1.average_std[1].item()}")
        print(f"Loaded ln_1 bos_std[0]={loaded_block0_ln1.bos_std[0].item()}, bos_std[1]={loaded_block0_ln1.bos_std[1].item()}")
        print(f"Loaded ln_2 values: is_fake={loaded_block0_ln2.is_fake}, real_average_std={loaded_block0_ln2.real_average_std}")

        # Check if attributes are preserved for ln_1
        print("Verifying ln_1 attributes...")
        assert loaded_block0_ln1.is_fake == block0_ln1.is_fake, "is_fake not preserved"
        assert loaded_block0_ln1.attn_v_is_fake == block0_ln1.attn_v_is_fake, "attn_v_is_fake not preserved"
        assert abs(loaded_block0_ln1.real_average_std - block0_ln1.real_average_std) < 1e-5, "real_average_std not preserved"
        assert abs(loaded_block0_ln1.real_bos_std - block0_ln1.real_bos_std) < 1e-5, "real_bos_std not preserved"
        
        # Compare tensor values using item() to avoid device issues
        assert abs(loaded_block0_ln1.average_std[0].item() - block0_ln1.average_std[0].item()) < 1e-5, "average_std[0] not preserved"
        assert abs(loaded_block0_ln1.average_std[1].item() - block0_ln1.average_std[1].item()) < 1e-5, "average_std[1] not preserved"
        assert abs(loaded_block0_ln1.bos_std[0].item() - block0_ln1.bos_std[0].item()) < 1e-5, "bos_std[0] not preserved"
        assert abs(loaded_block0_ln1.bos_std[1].item() - block0_ln1.bos_std[1].item()) < 1e-5, "bos_std[1] not preserved"

        # Check ln_2
        print("Verifying ln_2 attributes...")
        assert loaded_block0_ln2.is_fake == block0_ln2.is_fake, "ln_2 is_fake not preserved"
        assert abs(loaded_block0_ln2.real_average_std - block0_ln2.real_average_std) < 1e-5, "ln_2 real_average_std not preserved"

        # Check final ln
        print("Verifying final ln attributes...")
        assert loaded_final_ln.is_fake == final_ln.is_fake, "final ln is_fake not preserved"
        assert abs(loaded_final_ln.real_bos_std - final_ln.real_bos_std) < 1e-5, "final ln real_bos_std not preserved"

        print("FakeLayerNorm serialization test passed!")


if __name__ == "__main__":
    print("Running FakeLayerNorm tests...")
    test_fake_ln()
    test_serialization()
