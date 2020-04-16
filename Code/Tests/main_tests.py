import tensorly as tl
import math
import numpy as np

from Code.Models.Main import Model

#We test the RSE method using vectors, matrices and 3D tensors
model = Model()

#Vector test
vect = tl.tensor(np.array([2,2]))
prediction = tl.tensor(np.array([1,2]))
model.set_data(vect)

assert model.evaluate(vect, metric="RSE") == 0
assert model.evaluate(prediction, metric="RSE") == math.sqrt((1/8))

#Matrice test
matrice = tl.tensor(np.array([[1,2],
                              [3,4]]))
prediction = tl.tensor(np.array([[-1,1],
                                [3,2]]))
model.set_data(matrice)

assert model.evaluate(matrice, metric="RSE") == 0
assert model.evaluate(prediction, metric="RSE") == 3 / math.sqrt(30)

#3D tensor test
tensor = tl.tensor(np.array([[[1, 5],
                              [2, 6]],
                              [[3, 7],
                               [4, 8]]]))
prediction = tl.tensor(np.array([[[1, 3],
                                 [2, 6]],
                                [[4, 7],
                                [2, 7]]]))
model.set_data(tensor)

assert model.evaluate(tensor, metric="RSE") == 0
assert model.evaluate(prediction, metric="RSE") == math.sqrt(10/204)