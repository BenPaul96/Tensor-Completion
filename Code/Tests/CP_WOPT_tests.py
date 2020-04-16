import tensorly as tl
import numpy as np

from Code.Models.CP_WOPT import CP_WOPT_Model

#Test W, Y, Gamma
rank = 2
lr = 0.1

tensor = tl.tensor([[[1,2], [np.nan, 4]], [[np.nan, 6], [7, 8]]])
model = CP_WOPT_Model(tensor, rank, lr)

true_W = [[[1,1], [0,1]], [[0,1], [1,1]]]
true_Y = [[[1,2], [0, 4]], [[0, 6], [7, 8]]]
true_gamma = 170
assert np.array_equal(model.W, true_W)
assert np.array_equal(model.Y, true_Y)
assert true_gamma == model.gamma

#Test forward function
model = CP_WOPT_Model(tensor, rank, lr)

A0 = tl.tensor([[1,3], [2,4]]).astype(np.float)
A1 = tl.tensor([[2,1], [1,2]]).astype(np.float)
A2 = tl.tensor([[3,1], [2,2]]).astype(np.float)

model.factors = {"A0":A0, "A1":A1, "A2":A2}
true_Z = [[[9,10], [0,14]], [[0,16], [14, 20]]]
true_loss = 260.5

loss = model.forward()
assert np.array_equal(model.Z, true_Z)
assert loss == true_loss

#Test backward function
model.backward()

assert np.array_equal(model.grads["G0"], [[100, 64], [85, 82]])
assert np.array_equal(model.grads["G1"], [[80, 152], [110, 184]])
assert np.array_equal(model.grads["G2"], [[30, 80], [90, 220]])

#Test update function
model.update()

assert np.allclose(model.factors["A0"], [[-9. , -3.4],[-6.5, -4.2]])
assert np.allclose(model.factors["A1"], [[-6. , -14.2],[-10, -16.4]])
assert np.allclose(model.factors["A2"], [[0, -7],[-7, -20]])

#Test train function
n_epoch = 500
model.lr = 0.00002
model.train(n_epoch)
print(model.train_logs[f"Epoch{n_epoch-1}"])
model.predict()