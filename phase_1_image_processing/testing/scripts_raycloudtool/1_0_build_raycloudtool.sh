#!/bin/bash
# filepath: /media/laura/laura_usb/code/fvc_structure/phase_1_image_processing/step_1_imagesAlignment/scripts_raycloudtool/1_0_build_raycloudtool.sh

# Remove set -e to prevent immediate exit
# set -e  

echo "🔧 Building Raycloud Tools with LAS Support..."
echo "=================================================="

# Navigate to raycloudtools directory
echo "📁 Navigating to raycloudtools directory..."
cd /media/laura/laura_usb/code/fvc_structure/raycloudtools
pwd

# Fix repository issues first
echo ""
echo "🔧 Fixing repository issues..."
sudo apt-get clean
sudo apt-get update --allow-releaseinfo-change 2>/dev/null || true

# Install PDAL instead of liblas (more modern and available)
echo ""
echo "📦 Installing PDAL (modern LAS library)..."
sudo apt-get install -y pdal libpdal-dev libpdal-base-dev || {
    echo "⚠️  PDAL installation failed, trying basic dependencies..."
    sudo apt-get install -y cmake build-essential libeigen3-dev
}

# Install additional dependencies that might be needed
echo ""
echo "📦 Installing build dependencies..."
sudo apt-get install -y cmake build-essential libeigen3-dev libpcl-dev pkg-config || {
    echo "⚠️  Some packages failed to install, continuing..."
}

# Remove old build to ensure clean rebuild
echo ""
echo "🧹 Removing old build directory..."
if [ -d "build" ]; then
    rm -rf build/
    echo "   ✅ Old build directory removed"
fi

# Create new build directory
echo ""
echo "📁 Creating fresh build directory..."
mkdir build
cd build
echo "   ✅ Build directory created: $(pwd)"

# Try building WITHOUT LAS support first (to see if basic build works)
echo ""
echo "⚙️  First attempting build WITHOUT LAS support..."
cmake .. -DCMAKE_BUILD_TYPE=Release

CMAKE_EXIT_CODE=$?
if [ $CMAKE_EXIT_CODE -eq 0 ]; then
    echo "   ✅ Basic CMake configuration successful"
    
    # Try to build basic version
    echo "   🔨 Testing basic build..."
    make -j2  # Use fewer cores to avoid overwhelming
    
    MAKE_EXIT_CODE=$?
    if [ $MAKE_EXIT_CODE -eq 0 ]; then
        echo "   ✅ Basic build successful!"
        
        # Now try to rebuild with LAS support
        echo ""
        echo "⚙️  Now trying to add LAS support..."
        make clean
        cd ..
        rm -rf build
        mkdir build
        cd build
        
        # Try with LAS support
        cmake .. -DWITH_LAS=true -DCMAKE_BUILD_TYPE=Release
        
        if [ $? -eq 0 ]; then
            echo "   ✅ LAS-enabled CMake configuration successful"
            make -j2
            
            if [ $? -eq 0 ]; then
                echo "   ✅ LAS-enabled build successful!"
                LAS_SUPPORT=true
            else
                echo "   ⚠️  LAS build failed, using basic build..."
                # Rebuild without LAS
                cd ..
                rm -rf build
                mkdir build
                cd build
                cmake .. -DCMAKE_BUILD_TYPE=Release
                make -j2
                LAS_SUPPORT=false
            fi
        else
            echo "   ⚠️  LAS configuration failed, using basic build..."
            LAS_SUPPORT=false
        fi
    else
        echo "   ❌ Basic build failed!"
        exit 1
    fi
else
    echo "   ❌ Basic CMake configuration failed!"
    echo "   Checking for missing dependencies..."
    
    # Show what's available
    echo "   Installed cmake version:"
    cmake --version || echo "   cmake not found"
    
    echo "   Checking for compiler:"
    gcc --version || echo "   gcc not found"
    
    exit 1
fi

# Check what was built
echo ""
echo "📋 Checking built executables..."
if [ -d "bin" ]; then
    echo "   Contents of bin/ directory:"
    ls -la bin/ | head -15
    
    # Test rayimport
    if [ -f "bin/rayimport-0.0.1" ]; then
        echo ""
        echo "🧪 Testing rayimport..."
        
        # Show help to see if it mentions LAS support
        echo "   rayimport help output:"
        ./bin/rayimport-0.0.1 --help
        
        if [ "$LAS_SUPPORT" = true ]; then
            echo "   ✅ Built with LAS support enabled"
        else
            echo "   ⚠️  Built without LAS support - will need PLY conversion"
        fi
    else
        echo "   ❌ rayimport-0.0.1 not found"
    fi
else
    echo "   ❌ bin/ directory not found"
    echo "   Contents of current directory:"
    ls -la
fi

echo ""
echo "🎯 Build process complete!"

if [ "$LAS_SUPPORT" = true ]; then
    echo "   ✅ LAS support: ENABLED"
    echo "   📋 You can use .las files directly:"
    echo "      ./bin/rayimport-0.0.1 yourfile.las trajectory.txt"
else
    echo "   ⚠️  LAS support: DISABLED"
    echo "   📋 You'll need to convert LAS to PLY first:"
    echo "      pdal translate input.las output.ply"
    echo "      ./bin/rayimport-0.0.1 output.ply trajectory.txt"
fi

echo ""
echo "   Build directory: $(pwd)"
echo "   Next step: Test with your files!"

##############
# 🧪 Testing rayimport...
#    rayimport help output:
# Import a point cloud and trajectory file into a ray cloud
# usage:
# rayimport pointcloudfile trajectoryfile  - pointcloudfile can be a .laz, .las or .ply file
#                                            trajectoryfile is a text file using 'time x y z' format per line
# rayimport pointcloudfile 0,0,0           - use 0,0,0 as the sensor location
# rayimport pointcloudfile ray 0,0,-10     - use 0,0,-10 as the constant ray vector from start to point
#                                         --max_intensity 100 - specify maximum intensity value (default 100).
#                                                               0 sets all to full intensity (bounded rays).
#                                         --remove_start_pos  - translate so first point is at 0,0,0
# The output is a .ply file of the same name (or with suffix _raycloud if the input was a .ply file).
#    ✅ Built with LAS support enabled

# 🎯 Build process complete!
#    ✅ LAS support: ENABLED
#    📋 You can use .las files directly:
#       ./bin/rayimport-0.0.1 yourfile.las trajectory.txt

#    Build directory: /media/laura/laura_usb/code/fvc_structure/raycloudtools/build
#    Next step: Test with your files!