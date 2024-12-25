lis1 = [1, 3, 5, 7, 9]
lis2 = [8, 6, 4, 2, 15]
lis1.extend(lis2)
# lis1 += lis2
lis1.sort()
lis1.reverse()
# lis1.sort(reverse=True)
print(lis1)

print(len(lis1))
print(max(lis1))
print(min(lis1))
print(sum(lis1))
