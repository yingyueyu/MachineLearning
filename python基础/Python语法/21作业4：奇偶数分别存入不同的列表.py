l2 = [23, 45, 66, 82, 97, 120, 135, 140]
A = []
B = []
for i in l2:
    if i % 2 == 0:
        B.append(i)
    else:
        A.append(i)
print(A, B)