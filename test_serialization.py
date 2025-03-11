import os
import torch
import torch.nn.functional as F
from train import load_model, FakeLayerNorm, replace_layernorm_with_fake_layernorm
import shutil
from transformers import AutoModelForCausalLM

def test_fakelayernorm_serialization():
    """
    Test that FakeLayerNorm state (mode, attn_v_mode) is preserved
    during model serialization and deserialization.
    """
    print("=== FakeLayerNorm Serialization Test ===")
    
    # Directory for saving models
    test_dir = "test_models"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)  # Clean up before test
    
    os.makedirs(test_dir, exist_ok=True)
    save_path = os.path.join(test_dir, "test_fakelayernorm")
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Load a model with FakeLayerNorm
    print("\n1. Loading and initializing model...")
    model = load_model("gpt2", remove_ln=True)
    
    # 2. Set modes explicitly for testing
    print("\n2. Setting modes for testing...")
    block0_ln1_mode = "fake"
    block1_ln1_mode = "real" 
    block2_ln2_mode = "fake"
    ln_f_mode = "fake"
    
    print(f"Setting h.0.ln_1.mode = {block0_ln1_mode}")
    print(f"Setting h.1.ln_1.mode = {block1_ln1_mode}")
    print(f"Setting h.2.ln_2.mode = {block2_ln2_mode}")
    print(f"Setting ln_f.mode = {ln_f_mode}")
    
    model.transformer.h[0].ln_1.mode = block0_ln1_mode
    model.transformer.h[1].ln_1.mode = block1_ln1_mode
    model.transformer.h[2].ln_2.mode = block2_ln2_mode
    model.transformer.ln_f.mode = ln_f_mode
    
    # Check that buffer values are properly updated by the setters
    print("\nVerifying buffer values were updated by setters:")
    print(f"h.0.ln_1.mode_buffer = {model.transformer.h[0].ln_1.mode_buffer.item()} (should be 1 for fake)")
    print(f"h.1.ln_1.mode_buffer = {model.transformer.h[1].ln_1.mode_buffer.item()} (should be 0 for real)")
    print(f"h.2.ln_2.mode_buffer = {model.transformer.h[2].ln_2.mode_buffer.item()} (should be 1 for fake)")
    print(f"ln_f.mode_buffer = {model.transformer.ln_f.mode_buffer.item()} (should be 1 for fake)")
    
    # 3. Save the model
    print("\n3. Saving model...")
    # Check state_dict before saving
    state_dict = model.state_dict()
    mode_buffer_keys = [k for k in state_dict.keys() if "mode_buffer" in k]
    print(f"Found {len(mode_buffer_keys)} mode_buffer keys in state_dict")
    
    for key in mode_buffer_keys[:10]:  # Print first 10 keys
        print(f"  {key} = {state_dict[key].item()}")
    
    # Save with both formats for testing
    # 1. Traditional PyTorch saving (recommended)
    torch_save_path = os.path.join(save_path, "pytorch_model.bin")
    print(f"Saving PyTorch model to {torch_save_path}")
    torch.save(model.state_dict(), torch_save_path)
    
    # 2. Also save model config
    config_path = os.path.join(save_path, "config.json")
    if not os.path.exists(config_path):
        # Create a simple config
        import json
        config = {"model_type": "gpt2", "contains_fakelayernorm": True}
        with open(config_path, "w") as f:
            json.dump(config, f)
    
    print(f"Model saved to {save_path}")
    
    # 4. Load the model again
    print("\n4. Loading model from checkpoint...")
    loaded_model = load_model("gpt2", remove_ln=True, checkpoint_path=save_path)
    
    # Check buffer values in loaded model
    print("\nBuffer values in loaded model:")
    print(f"h.0.ln_1.mode_buffer = {loaded_model.transformer.h[0].ln_1.mode_buffer.item()}")
    print(f"h.1.ln_1.mode_buffer = {loaded_model.transformer.h[1].ln_1.mode_buffer.item()}")
    print(f"h.2.ln_2.mode_buffer = {loaded_model.transformer.h[2].ln_2.mode_buffer.item()}")
    print(f"ln_f.mode_buffer = {loaded_model.transformer.ln_f.mode_buffer.item()}")
    
    # 5. Verify that the modes were preserved
    print("\n5. Verifying modes were preserved...")
    success = True
    
    # Check specific modes we set
    tests = [
        (model.transformer.h[0].ln_1.mode, loaded_model.transformer.h[0].ln_1.mode, "h.0.ln_1.mode"),
        (model.transformer.h[1].ln_1.mode, loaded_model.transformer.h[1].ln_1.mode, "h.1.ln_1.mode"),
        (model.transformer.h[2].ln_2.mode, loaded_model.transformer.h[2].ln_2.mode, "h.2.ln_2.mode"),
        (model.transformer.ln_f.mode, loaded_model.transformer.ln_f.mode, "ln_f.mode"),
    ]
    
    for original, loaded, name in tests:
        if original == loaded:
            print(f"✅ {name} preserved: {original}")
        else:
            print(f"❌ {name} mismatch: Original={original}, Loaded={loaded}")
            success = False
    
    # Final result
    if success:
        print("\n✅ Test passed: FakeLayerNorm serialization and deserialization work correctly!")
        # Clean up test directory
        shutil.rmtree(test_dir)
        print(f"Test directory {test_dir} cleaned up")
    else:
        print("\n❌ Test failed: Some FakeLayerNorm states were not preserved.")
        print(f"\nTest directory preserved for inspection: {test_dir}")
    
    return success

if __name__ == "__main__":
    test_fakelayernorm_serialization() 