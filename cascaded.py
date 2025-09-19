import scipy.io

# Replace 'your_file.mat' with the path to your .mat file
mat = scipy.io.loadmat('/Users/hakilic/Downloads/Tensor-Network-B-splines-master/Cascaded/dataBenchmark.mat')

# Print keys to see what’s inside
print(mat.keys())
