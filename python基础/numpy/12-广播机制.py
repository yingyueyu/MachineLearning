import numpy as np

a = np.array([[0, 0, 0],
              [10, 10, 10],
              [20, 20, 20],
              [30, 30, 30]
              ])
b = np.array([0, 1, 2])
# tile() 扩展数组， 4：把行重复4遍， 1保持列不变
# b = np.tile(b, (4, 1))
print(b)

print(a + b)

a = np.array([
    [
        [2, 3]
    ],
    [
        [2, 3]
    ],
    [
        [2, 3]
    ],
    [
        [2, 3]
    ]
])
b = np.array([
    [
        [2, 3],
        [1, 2],
        [4, 3]
    ]
])
print(a.shape)
print(b.shape)
print(a + b)
