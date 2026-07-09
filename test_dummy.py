import torch
import numpy as np

# Import models
from models.TinySleepNet.models.tinysleepnet import TinySleepNet
from models.DeepSleepNet.deepsleepnet import DeepSleepNet
from models.SleepTransformer.sleeptransformer import SleepTransformer
from models.MambaSleep.mambasleep import MambaSleep

# Import dataset functions to test
from common.dataset import temporal_shift_no_wrap

def test_temporal_shift():
    print("Testing temporal_shift_no_wrap...")
    x = np.arange(10).reshape(1, 1, 10)
    
    # Test positive shift
    shifted_pos = temporal_shift_no_wrap(x, 2)
    expected_pos = np.array([[[0, 0, 0, 1, 2, 3, 4, 5, 6, 7]]])
    assert np.array_equal(shifted_pos, expected_pos), f"Positive shift failed: {shifted_pos}"
    
    # Test negative shift
    shifted_neg = temporal_shift_no_wrap(x, -2)
    expected_neg = np.array([[[2, 3, 4, 5, 6, 7, 8, 9, 0, 0]]])
    assert np.array_equal(shifted_neg, expected_neg), f"Negative shift failed: {shifted_neg}"
    
    # Test zero shift
    shifted_zero = temporal_shift_no_wrap(x, 0)
    assert np.array_equal(shifted_zero, x), f"Zero shift failed: {shifted_zero}"
    
    print("temporal_shift_no_wrap works perfectly!")

def test_models():
    # Input shape: (B, L, C, Length) -> (2, 20, 1, 3000)
    B, L, C, Length = 2, 20, 1, 3000
    x = torch.randn(B, L, C, Length)
    mask = torch.ones(B, L, dtype=torch.bool)
    
    models = {
        "TinySleepNet": TinySleepNet(in_channels=1, num_classes=5),
        "DeepSleepNet": DeepSleepNet(in_channels=1, num_classes=5),
        "SleepTransformer": SleepTransformer(in_channels=1, num_classes=5),
        "MambaSleep": MambaSleep(in_channels=1, num_classes=5)
    }
    
    for name, model in models.items():
        print(f"Running forward for {name}...")
        try:
            out = model(x, mask=mask)
            print(f"{name} output shape: {out.shape}")
            assert out.shape == (2, 20, 5), f"{name} shape is {out.shape}, expected (2, 20, 5)"
        except Exception as e:
            print(f"Error running {name}: {e}")
            raise e

if __name__ == "__main__":
    test_temporal_shift()
    test_models()
    print("All tests passed successfully!")
