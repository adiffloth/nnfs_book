import numpy as np

inputs = np.array([[1.0, 2.0, 3.0, 2.5],
                   [2.0, 5.0, -1.0, 2.0],
                   [-1.5, 2.7, 3.3, -0.8]])

weights = np.array([[0.2, 0.8, -0.5, 1.0],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

biases = np.array([2.0, 3.0, 0.5])

print(f'inputs: {inputs.shape}')
print(f'weights: {weights.shape}')
print(f'biases: {biases.shape}')
layer_outputs = np.dot(inputs, weights) + biases
print(layer_outputs)
print(f'layer_outputs: {layer_outputs.shape}')

expected = np.array([[4.8, 1.21, 2.385],
                     [8.9, -1.81, 0.2],
                     [1.41, 1.051, 0.026]])

assert np.allclose(layer_outputs, expected), "The output does not match the expected values."

'''
array([[ 4.8    1.21   2.385],
       [ 8.9   -1.81   0.2  ],
       [ 1.41   1.051  0.026]])
'''

print('---')
print(inputs)
print()
print(weights)
print('---')
print('expected:')
print(np.dot(inputs, weights))
print('---')


def dot_prod(a, b):
    if a.shape[0] != b.shape[1]:
        raise ValueError()
    out = []
    for i in range(a.shape[0]):
        r = []
        for j in range(b.shape[1]):
            r.append(sum([a*b for a, b in zip(a[i, :], b[:, j])]))
        out.append(r)
    return np.array(out)


print('dot_prod:')
print(dot_prod(inputs, weights))
assert np.allclose(dot_prod(inputs, weights), np.dot(inputs, weights))
print('All tests pass.')
