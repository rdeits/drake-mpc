# Installation

    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=install
    make

# Where are the drake files?

After you've run the installation steps, you can find the drake folder in:

    build/drake-prefix/drake

# How to update the version of drake we're using?

Edit the `CMakeLists.txt` file and change the `GIT_TAG` variable to the SHA of the commit that you want.

