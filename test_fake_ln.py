import torch
from train import FakeLayerNorm
from std_dicts import std_dicts

def test_fake_ln():
    ln = FakeLayerNorm(n_embd=4, n_ctx=2, layer="blocks.0.hook_resid_pre", bias=True, init_average_std=0.2, init_bos_std=2.0)

    # Test initialization - using property access for scalar values
    assert abs(ln.real_average_std - 0.2) < 1e-5, f"Expected real_average_std to be 0.2, got {ln.real_average_std}"
    assert abs(ln.real_bos_std - 2.0) < 1e-5, f"Expected real_bos_std to be 2.0, got {ln.real_bos_std}"
    
    # Test that the underlying buffer values are correct
    assert torch.allclose(ln._real_average_std, torch.tensor(0.2, device=ln._real_average_std.device))
    assert torch.allclose(ln._real_bos_std, torch.tensor(2.0, device=ln._real_bos_std.device))
    
    # Get expected values based on initialization
    expected_avg_std = torch.ones(4, device=ln.average_std.device) * 0.2
    expected_avg_std[0] = 2.0
    expected_bos_std = torch.ones(4, device=ln.bos_std.device) * 2.0
    
    # Test the tensor buffer values with proper device handling
    assert torch.allclose(ln.average_std, expected_avg_std), f"Expected average_std to be {expected_avg_std}, got {ln.average_std}"
    assert torch.allclose(ln.bos_std, expected_bos_std), f"Expected bos_std to be {expected_bos_std}, got {ln.bos_std}"
    
    # Test flag property getters and setters
    assert ln.is_fake == False, f"Expected is_fake to be False, got {ln.is_fake}"
    ln.is_fake = True
    assert ln.is_fake == True, f"Expected is_fake to be True after setting, got {ln.is_fake}"
    assert ln._is_fake.item() == True, f"Expected _is_fake buffer to be True, got {ln._is_fake.item()}"
    
    assert ln.attn_v_is_fake == False, f"Expected attn_v_is_fake to be False, got {ln.attn_v_is_fake}"
    ln.attn_v_is_fake = True
    assert ln.attn_v_is_fake == True, f"Expected attn_v_is_fake to be True after setting, got {ln.attn_v_is_fake}"
    assert ln._attn_v_is_fake.item() == True, f"Expected _attn_v_is_fake buffer to be True, got {ln._attn_v_is_fake.item()}"
    
    assert ln.bos_special_treatment == True, f"Expected bos_special_treatment to be True, got {ln.bos_special_treatment}"
    ln.bos_special_treatment = False
    assert ln.bos_special_treatment == False, f"Expected bos_special_treatment to be False after setting, got {ln.bos_special_treatment}"
    assert ln._bos_special_treatment.item() == False, f"Expected _bos_special_treatment buffer to be False, got {ln._bos_special_treatment.item()}"
    
    print("Basic FakeLayerNorm test passed!")


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
        # Modify attributes using property setters (which update the underlying buffers)
        block0_ln1.is_fake = True
        block0_ln1.attn_v_is_fake = True
        block0_ln1.real_average_std = 0.5
        block0_ln1.real_bos_std = 2.5
        block0_ln1.bos_special_treatment = False
        
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

        # Print initial values - read from properties 
        print(f"Original ln_1 values: is_fake={block0_ln1.is_fake}, attn_v_is_fake={block0_ln1.attn_v_is_fake}")
        print(f"Original ln_1 bos_special_treatment={block0_ln1.bos_special_treatment}")
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
        
        # Debug: Print buffer keys in state dict related to ln_1
        print("\nDebug - State dict keys related to ln_1:")
        buffer_keys = []
        for key in [k for k in state_dict.keys() if "ln_1" in k]:
            print(f"  {key}")
            if "_is_fake" in key or "_real_average_std" in key or "average_std" in key:
                buffer_keys.append(key)
        
        if not buffer_keys:
            print("\nWARNING: No buffer keys found in state_dict! This suggests the buffers aren't being properly saved.")
            
        print("")
            
        loaded_model.load_state_dict(state_dict)
        print("Model loaded successfully.")

        # Get the loaded versions of the components
        loaded_block0_ln1 = loaded_model.transformer.h[0].ln_1
        loaded_block0_ln2 = loaded_model.transformer.h[0].ln_2
        loaded_final_ln = loaded_model.transformer.ln_f
        
        # Print loaded values - using properties to access the underlying buffers
        print(f"Loaded ln_1 values: is_fake={loaded_block0_ln1.is_fake}, attn_v_is_fake={loaded_block0_ln1.attn_v_is_fake}")
        print(f"Loaded ln_1 bos_special_treatment={loaded_block0_ln1.bos_special_treatment}")
        print(f"Loaded ln_1 std values: real_avg={loaded_block0_ln1.real_average_std}, real_bos={loaded_block0_ln1.real_bos_std}")
        print(f"Loaded ln_1 average_std[0]={loaded_block0_ln1.average_std[0].item()}, average_std[1]={loaded_block0_ln1.average_std[1].item()}")
        print(f"Loaded ln_1 bos_std[0]={loaded_block0_ln1.bos_std[0].item()}, bos_std[1]={loaded_block0_ln1.bos_std[1].item()}")
        print(f"Loaded ln_2 values: is_fake={loaded_block0_ln2.is_fake}, real_average_std={loaded_block0_ln2.real_average_std}")

        # Check if attributes are preserved for ln_1
        print("Verifying ln_1 attributes...")
        assert loaded_block0_ln1.is_fake == block0_ln1.is_fake, "is_fake not preserved"
        assert loaded_block0_ln1.attn_v_is_fake == block0_ln1.attn_v_is_fake, "attn_v_is_fake not preserved"
        assert loaded_block0_ln1.bos_special_treatment == block0_ln1.bos_special_treatment, "bos_special_treatment not preserved"
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
