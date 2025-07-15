#!/usr/bin/env python3
"""
Unit test to verify SPOC's old ai2thor version can work with different rendering solutions.
This test will help determine if Xvfb or other approaches can solve the rendering issue.
"""

import os
import sys
import subprocess
import time
import traceback

def test_rendering_configuration(config_name, setup_func):
    """Test a specific rendering configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print('='*60)
    
    try:
        # Run setup
        setup_func()
        
        # Try to import and initialize ai2thor
        import ai2thor.controller
        
        print(f"✓ Successfully imported ai2thor")
        
        # Test different platform configurations
        test_configs = [
            {
                "name": "Linux64 with headless",
                "config": {
                    "platform": "Linux64",
                    "headless": True,
                    "width": 512,
                    "height": 512,
                    "agentMode": "stretch",
                    "gridSize": 0.1,
                    "server_timeout": 30,
                }
            },
            {
                "name": "CloudRendering (should fail with old version)",
                "config": {
                    "platform": "CloudRendering", 
                    "headless": True,
                    "width": 512,
                    "height": 512,
                    "agentMode": "stretch",
                    "gridSize": 0.1,
                    "server_timeout": 30,
                }
            }
        ]
        
        for test in test_configs:
            print(f"\n  Testing platform: {test['name']}")
            controller = None
            try:
                controller = ai2thor.controller.Controller(**test['config'])
                print(f"    ✓ Controller created successfully")
                
                # Try to reset to a scene
                event = controller.reset(scene="FloorPlan1")
                if event and event.metadata.get('lastActionSuccess'):
                    print(f"    ✓ Scene reset successful")
                    print(f"    ✓ Frame shape: {event.frame.shape if hasattr(event, 'frame') else 'No frame'}")
                else:
                    print(f"    ✗ Scene reset failed")
                    
            except Exception as e:
                print(f"    ✗ Failed: {str(e)}")
            finally:
                if controller:
                    try:
                        controller.stop()
                    except:
                        pass
                        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {str(e)}")
        traceback.print_exc()
        return False

def setup_no_display():
    """Setup for testing without display."""
    os.environ['DISPLAY'] = ''
    print("Environment: DISPLAY unset")

def setup_xvfb_display():
    """Setup for testing with Xvfb virtual display."""
    # Check if Xvfb is available
    try:
        subprocess.run(['which', 'xvfb-run'], check=True, capture_output=True)
    except:
        print("⚠️  Xvfb not installed. Install with: sudo apt-get install xvfb")
        return
        
    os.environ['DISPLAY'] = ':99'
    print("Environment: DISPLAY=:99 (for Xvfb)")

def setup_headless_vars():
    """Setup environment variables for headless rendering."""
    env_vars = {
        'DISPLAY': '',
        'LIBGL_ALWAYS_SOFTWARE': '1',
        'GALLIUM_DRIVER': 'softpipe',
        'MESA_GL_VERSION_OVERRIDE': '3.3',
    }
    for key, value in env_vars.items():
        os.environ[key] = value
    print(f"Environment: {env_vars}")

def main():
    print("SPOC AI2-THOR Rendering Test")
    print("="*60)
    print("This test will verify different rendering solutions for the old ai2thor version")
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check if we're in the vagen environment
    if 'vagen' not in sys.prefix:
        print("\n⚠️  WARNING: Not in 'vagen' conda environment!")
        print("Please run: conda activate vagen")
        return
    
    # Check if the specific ai2thor version is installed
    try:
        import ai2thor
        print(f"AI2-THOR version: {getattr(ai2thor, '__version__', 'Unknown')}")
    except ImportError:
        print("\n✗ ai2thor not installed!")
        print("Please install the specific version required by SPOC:")
        print("pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+5d0ab8ab8760eb584c5ae659c2b2b951cab23246")
        return
    
    # Run tests
    results = []
    
    # Test 1: No display (should fail on Linux64)
    results.append(("No Display", test_rendering_configuration("No Display (unset DISPLAY)", setup_no_display)))
    
    # Test 2: Headless environment variables
    results.append(("Headless Vars", test_rendering_configuration("Headless Environment Variables", setup_headless_vars)))
    
    # Test 3: Xvfb (if available)
    results.append(("Xvfb", test_rendering_configuration("Xvfb Virtual Display", setup_xvfb_display)))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name:20} {status}")
    
    # Recommendation
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    if any(success for _, success in results):
        print("✓ At least one rendering solution works!")
        print("\nTo use Xvfb in your training script:")
        print("1. Install Xvfb: sudo apt-get install xvfb")
        print("2. Run your server with: xvfb-run -a -s '-screen 0 1024x768x24' python -m vagen.server.server")
        print("\nThis creates a virtual display that satisfies the old ai2thor's Linux64 requirements.")
    else:
        print("✗ No rendering solution worked. You may need to:")
        print("1. Check if the old ai2thor version is correctly installed")
        print("2. Consider using a Docker container with proper X11 setup")
        print("3. Investigate patching ai2thor to support CloudRendering")

if __name__ == "__main__":
    main()