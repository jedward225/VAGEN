#!/bin/bash
# 调试远程环境的脚本

echo "=== 检查环境变量 ==="
echo "MESA_GL_VERSION_OVERRIDE=$MESA_GL_VERSION_OVERRIDE"
echo "LIBGL_ALWAYS_SOFTWARE=$LIBGL_ALWAYS_SOFTWARE"
echo "EGL_PLATFORM=$EGL_PLATFORM"
echo "GALLIUM_DRIVER=$GALLIUM_DRIVER"
echo "DISPLAY=$DISPLAY"
echo "SPOC_DATA_PATH=$SPOC_DATA_PATH"

echo -e "\n=== 检查 vulkaninfo ==="
echo "PATH=$PATH"
which vulkaninfo
vulkaninfo 2>&1 | head -5

echo -e "\n=== 检查 fake vulkaninfo 是否存在 ==="
ls -la ~/bin/vulkaninfo 2>/dev/null || echo "~/bin/vulkaninfo 不存在"

echo -e "\n=== 检查 OpenGL ==="
glxinfo 2>&1 | head -10 || echo "glxinfo 不可用"

echo -e "\n=== 检查数据路径 ==="
ls -la /root/spoc_data/fifteen/ 2>/dev/null || echo "/root/spoc_data/fifteen/ 不存在"

echo -e "\n=== 设置环境变量并测试 ==="
# 设置所有必要的环境变量
export MESA_GL_VERSION_OVERRIDE=3.3
export LIBGL_ALWAYS_SOFTWARE=1
export EGL_PLATFORM=surfaceless
export GALLIUM_DRIVER=llvmpipe
export SPOC_DATA_PATH=/root/spoc_data/fifteen

# 创建 fake vulkaninfo
mkdir -p ~/bin
cat > ~/bin/vulkaninfo << 'EOF'
#!/bin/bash
echo "Fake vulkaninfo"
echo "Vulkan Instance Version: 1.2.170"
echo ""
echo "Devices:"
echo "========"
echo "GPU0:"
echo "    deviceUUID = cc0a9986-229c-fdb6-d361-b0c69aab1d62"
exit 0
EOF
chmod +x ~/bin/vulkaninfo
export PATH=~/bin:$PATH

# 删除 DISPLAY
unset DISPLAY

echo -e "\n=== 重新检查环境变量 ==="
echo "MESA_GL_VERSION_OVERRIDE=$MESA_GL_VERSION_OVERRIDE"
echo "LIBGL_ALWAYS_SOFTWARE=$LIBGL_ALWAYS_SOFTWARE"
echo "vulkaninfo 输出:"
vulkaninfo | head -10

echo -e "\n=== 测试 Python 导入 ==="
python3 -c "
import os
print('Python 环境变量:')
for var in ['MESA_GL_VERSION_OVERRIDE', 'LIBGL_ALWAYS_SOFTWARE', 'EGL_PLATFORM', 'GALLIUM_DRIVER', 'DISPLAY']:
    print(f'  {var}={os.environ.get(var, \"NOT SET\")}')

try:
    from ai2thor.controller import Controller
    print('\\nAI2-THOR 导入成功')
except Exception as e:
    print(f'\\nAI2-THOR 导入失败: {e}')
"

echo -e "\n=== 完成 ==="