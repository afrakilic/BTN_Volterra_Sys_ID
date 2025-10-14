import scipy.io
from lagfeatures import lagfeatures
from config import *  # Import everything from config.py
from volterra_BTN import btnkm
# Replace 'your_file.mat' with the path to your .mat file
mat = scipy.io.loadmat('/Users/hakilic/Downloads/Tensor-Network-B-splines-master/Cascaded/dataBenchmark.mat')

uEst = mat['uEst'].squeeze()
uVal = mat['uVal'].squeeze()
yEst = mat['yEst'].squeeze()
yVal = mat['yVal'].squeeze()

#Normalize the input and output training data to [0 1]
u_train = uEst.reshape(-1, 1) #/7
y_train = yEst#/10 - 0.1
u_test = uVal.reshape(-1, 1) #/ 7
y_test = yVal#/10 - 0.1


X_mean = u_train.mean(axis=0)
X_std = u_train.std(axis=0)
X_std[X_std == 0] = 1
X_train = (u_train - X_mean) / X_std
X_test = (u_test - X_mean) / X_std  # Use train stats

y_mean = y_train.mean()
y_std = y_train.std()
y_train = (y_train - y_mean) / y_std


# hyper-parameters
input_dimension = 100
max_rank = 20
Kernel_Degree = 3

a, b = 1e-4, 1e-3
c, d = 1e-5 * np.ones(max_rank), 1e-6 * np.ones(max_rank)
g, h = 1e-5 * np.ones(input_dimension+1), 1e-6 * np.ones(input_dimension+1)

model = btnkm(Kernel_Degree)
R, W_D, lambda_M, lambda_R = model.train(
        features=u_train,
        target=y_train,
        input_dimension=input_dimension,
        Volterra_Degree=Kernel_Degree, 
        max_rank=max_rank,
        shape_parameter_tau=a,
        scale_parameter_tau=b,
        shape_parameter_lambda_R=c,
        scale_parameter_lambda_R=d,
        shape_parameter_lambda_M=g,
        scale_parameter_lambda_M=h,
        max_iter=20,
        lower_bound_tol=1e-4,
        precision_update=True,
        lambda_R_update=True,
        lambda_M_update=True,
        plot_results=True,
        prune_rank=True,
    )


    # Predict (mse is returned by the predict function)
prediction_mean, prediction_std, _ = model.predict(
    features=u_test, input_dimension=input_dimension
)



plt.figure(figsize=(8, 5))
plt.plot(np.log(lambda_M), marker='o', linestyle='-', alpha=0.8)
plt.title('Ordered λ_M Values')
plt.xlabel('Sample Index (sorted)')
plt.ylabel('λ_M Value')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

prediction_mean_unscaled = prediction_mean[input_dimension:, ] * y_std + y_mean
prediction_std_unscaled = prediction_std[input_dimension:, ] * y_std

rmse = np.sqrt(np.mean((prediction_mean_unscaled - y_test[input_dimension:, ]) ** 2))

print(rmse)
# nll
nll = 0.5 * np.log(2 * np.pi * prediction_std_unscaled**2) + 0.5 * (
    (y_test[input_dimension:, ] - prediction_mean_unscaled) ** 2
) / (prediction_std_unscaled**2)

nll = np.mean(nll)

print(nll)
# Create a time vector for plotting
time_steps = np.arange(len(y_test[input_dimension:, ]))

plt.figure(figsize=(14, 6))

# Plot actual test values
plt.plot(time_steps, y_test[input_dimension:, ], label='Actual Output (y_test)', color='blue', linewidth=2)

# Plot predicted values
plt.plot(time_steps, prediction_mean_unscaled, label='Predicted Output', color='orange', linewidth=2)

# Optionally, add confidence intervals (±1 std)
plt.fill_between(
    time_steps,
    prediction_mean_unscaled - 3* prediction_std_unscaled,
    prediction_mean_unscaled + 3*prediction_std_unscaled,
    color='orange',
    alpha=0.2,
    label='Prediction ±3 std'
)

# Styling
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.title('Test Output vs Predictions', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Show plot
plt.show()
