# This is a part of bigbrainwarp which can be found in
# https://github.com/caseypaquola/BigBrainWarp/blob/master/scripts/compile_profiles.py
# Modified slightly (saving as npz instead of txt + using os.path.join)
# Import packages
import sys
import os
import numpy

# Define input arguments
out_dir = str(sys.argv[1])
num_surf = int(sys.argv[2])
print(num_surf)

# Load top surface to learn shape
tmp=numpy.loadtxt(os.path.join(out_dir, "1.txt"))

# load all the surfaces and construct profiles
profiles = numpy.zeros([num_surf, tmp.shape[0]])
for i in range(0,num_surf):
    profiles[i,:] = numpy.loadtxt(os.path.join(out_dir, str(i) + ".txt"))

numpy.savez_compressed(os.path.join(out_dir, "profiles.npz"), profiles=profiles)
