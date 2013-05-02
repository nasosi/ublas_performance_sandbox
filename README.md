ublas_performance_sandbox
=========================

A sandbox repo for ublas performance experiments.
To download:
-----------
git clone https://github.com/nasosi/ublas_performance_sandbox.git

To compile:
-----------
cd ublas_performance_sandbox

qmake

make

or:

cd ublas_performance_sandbox

g++ -DNDEBUG -O3 -std=c++0x main.cpp -o benchmarks

To run:
---------
./benchmarks
