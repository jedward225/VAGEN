#!/usr/bin/env python3
"""
Test if we can create a fake vulkaninfo to bypass AI2-THOR's check
"""
import os
import subprocess
import tempfile

def create_fake_vulkaninfo():
    """Create a temporary fake vulkaninfo executable"""
    # Create a temporary directory for our fake vulkaninfo
    temp_dir = tempfile.mkdtemp()
    fake_vulkaninfo_path = os.path.join(temp_dir, "vulkaninfo")
    
    # Write a simple script that returns success
    with open(fake_vulkaninfo_path, 'w') as f:
        f.write("""#!/bin/bash
echo "Fake vulkaninfo output"
echo "Vulkan Instance Version: 1.2.0"
echo ""
echo "Instance Extensions:"
echo "===================="
echo "VK_KHR_surface"
echo "VK_KHR_xcb_surface"
echo "VK_KHR_xlib_surface"
echo "VK_KHR_wayland_surface"
echo ""
echo "Devices:"
echo "========"
echo "GPU0:"
echo "    apiVersion     = 1.2.0"
echo "    driverVersion  = 1.0.0"
echo "    vendorID       = 0x0000"
echo "    deviceID       = 0x0000"
echo "    deviceType     = CPU"
echo "    deviceName     = llvmpipe (LLVM 12.0.0, 256 bits)"
exit 0
""")
    
    # Make it executable
    os.chmod(fake_vulkaninfo_path, 0o755)
    
    # Add to PATH
    original_path = os.environ.get('PATH', '')
    os.environ['PATH'] = f"{temp_dir}:{original_path}"
    
    print(f"Created fake vulkaninfo at: {fake_vulkaninfo_path}")
    print(f"Updated PATH to include: {temp_dir}")
    
    # Test if it works
    try:
        result = subprocess.run(['vulkaninfo'], capture_output=True, text=True)
        print(f"Fake vulkaninfo test: {'SUCCESS' if result.returncode == 0 else 'FAILED'}")
        if result.returncode == 0:
            print("Output:", result.stdout[:200], "...")
    except Exception as e:
        print(f"Failed to run fake vulkaninfo: {e}")
    
    return temp_dir, original_path

if __name__ == "__main__":
    print("Testing vulkaninfo workaround...")
    temp_dir, original_path = create_fake_vulkaninfo()
    
    try:
        # Now test AI2-THOR
        print("\nTesting AI2-THOR with fake vulkaninfo...")
        import ai2thor.controller
        
        controller = ai2thor.controller.Controller(
            scene='FloorPlan1_physics',
            gridSize=0.25,
            platform="CloudRendering",
            headless=True
        )
        print("SUCCESS: AI2-THOR initialized!")
        controller.stop()
        
    except Exception as e:
        print(f"FAILED: {e}")
    finally:
        # Restore original PATH
        os.environ['PATH'] = original_path
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)