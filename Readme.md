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

# Using Drake from python

First, you need to make sure python knows how to find the Drake libraries. After you've followed the installation instructions, you just need to do (from this folder):

    source build/setup.sh

which will set the PYTHONPATH environment variable. You can test that it worked by running:

    python -m pydrake.test.testMathematicalProgram
