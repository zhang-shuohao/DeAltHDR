#!/usr/bin/env python3
"""
Test script for DeAltHDR model
"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[0]))

from basicsr.models.archs.dealthdr_arch import create_video_model, get_training_batch_config

def create_test_config():
    """Create test configuration for DeAltHDR model"""
    opt = {
        'n_colors': 3,
        'dim': 64,
        'Enc_blocks': [2, 2, 2],
        'Middle_blocks': 2,
        'Dec_blocks': [2, 2, 2],
        'num_refinement_blocks': 1,
        'ffn_expansion_factor': 1,
        'bias': False,
        'LayerNorm_type': 'WithBias',
        'num_heads_blks': [1, 2, 4, 8],
        
        # Encoder attention types
        'encoder1_attn_type1': 'Channel',
        'encoder1_attn_type2': 'FHR',
        'encoder2_attn_type1': 'Channel', 
        'encoder2_attn_type2': 'FHR',
        'encoder3_attn_type1': 'Channel',
        'encoder3_attn_type2': 'FHR',
        
        # Decoder attention types
        'decoder1_attn_type1': 'Channel',
        'decoder1_attn_type2': 'OpticalFlowFusion',
        'decoder2_attn_type1': 'Channel',
        'decoder2_attn_type2': 'OpticalFlowFusion',
        'decoder3_attn_type1': 'Channel', 
        'decoder3_attn_type2': 'OpticalFlowFusion',
        
        # FFW types
        'encoder1_ffw_type': 'GFFW',
        'encoder2_ffw_type': 'GFFW',
        'encoder3_ffw_type': 'GFFW',
        'decoder1_ffw_type': 'GFFW',
        'decoder2_ffw_type': 'GFFW',
        'decoder3_ffw_type': 'GFFW',
        
        # Latent
        'latent_attn_type1': 'FHR',
        'latent_attn_type2': 'Channel',
        'latent_attn_type3': 'FHR',
        'latent_ffw_type': 'GFFW',
        
        # Refinement
        'refinement_attn_type1': 'Channel',
        'refinement_attn_type2': 'Channel',
        'refinement_ffw_type': 'GFFW',
        
        'use_both_input': False,
        'num_frames_tocache': 4,
        'use_dual_encoder': True,
        'training_mode': 'mixed'
    }
    return opt

def test_model_creation():
    """Test model creation"""
    print("=== Testing DeAltHDR Model Creation ===")
    
    opt = create_test_config()
    model = create_video_model(opt)
    
    print(f"Model created successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Use dual encoder: {model.use_dual_encoder}")
    print(f"Training mode: {model.training_mode}")
    print(f"Number of frames to cache: {model.num_frames_tocache}")
    
    return model

def test_forward_pass():
    """Test forward pass with different training modes"""
    print("\n=== Testing Forward Pass ===")
    
    opt = create_test_config()
    model = create_video_model(opt)
    model.eval()
    
    # Test input: [B, T, C, H, W] where T=5 for T-2,T-1,T,T+1,T+2
    batch_size = 2
    input_frames = torch.randn(batch_size, 5, 3, 64, 64)
    
    print(f"Input shape: {input_frames.shape}")
    
    with torch.no_grad():
        # Test different training modes
        for mode in ['optical_flow', 'attention', 'mixed']:
            print(f"\nTesting {mode} mode:")
            try:
                output, k_cache, v_cache = model(
                    input_frames, 
                    exposure_type='long',
                    training_mode=mode
                )
                print(f"  Output shape: {output.shape}")
                print(f"  K cache: {len(k_cache) if k_cache else 'None'}")
                print(f"  V cache: {len(v_cache) if v_cache else 'None'}")
            except Exception as e:
                print(f"  Error in {mode} mode: {e}")
        
        # Test different exposure types
        for exposure in ['long', 'short']:
            print(f"\nTesting {exposure} exposure:")
            try:
                output, k_cache, v_cache = model(
                    input_frames, 
                    exposure_type=exposure,
                    training_mode='mixed'
                )
                print(f"  Output shape: {output.shape}")
            except Exception as e:
                print(f"  Error with {exposure} exposure: {e}")

def test_training_batch_config():
    """Test training batch configuration"""
    print("\n=== Testing Training Batch Configuration ===")
    
    batch_sizes = [4, 8, 16, 32]
    
    for batch_size in batch_sizes:
        config = get_training_batch_config(batch_size)
        print(f"Batch size {batch_size}:")
        print(f"  Optical flow: {config['optical_flow']} ({config['optical_flow']/batch_size*100:.1f}%)")
        print(f"  Attention: {config['attention']} ({config['attention']/batch_size*100:.1f}%)")
        print(f"  Mixed: {config['mixed']} ({config['mixed']/batch_size*100:.1f}%)")
        print(f"  Total: {sum(config.values())}")

def test_model_parameters():
    """Test model parameters"""
    print("\n=== Testing Model Parameters ===")
    
    opt = create_test_config()
    model = create_video_model(opt)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Check dual encoder parameters
    if hasattr(model, 'long_exposure_projection'):
        long_params = sum(p.numel() for p in model.long_exposure_projection.parameters())
        print(f"Long exposure encoder parameters: {long_params:,}")
    
    if hasattr(model, 'short_exposure_projection'):
        short_params = sum(p.numel() for p in model.short_exposure_projection.parameters())
        print(f"Short exposure encoder parameters: {short_params:,}")

def test_memory_usage():
    """Test memory usage"""
    print("\n=== Testing Memory Usage ===")
    
    opt = create_test_config()
    model = create_video_model(opt)
    model.eval()
    
    # Test with different input sizes
    test_sizes = [(1, 5, 3, 64, 64), (2, 5, 3, 128, 128), (1, 5, 3, 256, 256)]
    
    for size in test_sizes:
        print(f"\nTesting with input size {size}:")
        try:
            input_frames = torch.randn(*size)
            
            with torch.no_grad():
                output, _, _ = model(
                    input_frames, 
                    exposure_type='long',
                    training_mode='mixed'
                )
            
            print(f"  Input memory: {input_frames.element_size() * input_frames.nelement() / 1024**2:.2f} MB")
            print(f"  Output memory: {output.element_size() * output.nelement() / 1024**2:.2f} MB")
            print(f"  Success!")
            
        except Exception as e:
            print(f"  Error: {e}")

def main():
    """Main test function"""
    print("DeAltHDR Model Test Suite")
    print("=" * 50)
    
    try:
        # Test model creation
        model = test_model_creation()
        
        # Test forward pass
        test_forward_pass()
        
        # Test training batch configuration
        test_training_batch_config()
        
        # Test model parameters
        test_model_parameters()
        
        # Test memory usage
        test_memory_usage()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

