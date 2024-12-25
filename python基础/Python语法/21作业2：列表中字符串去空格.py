
s = ['123 45  67   9   ', '123', '  ab  c ']
# 方式二
# for i in range(len(s)):
#     if ' ' in s[i]:
#         s[i] = s[i].replace(' ', '')
# print(s)

# 方式二
# s1 = []
# for i in s:
#         s1.append(i.replace(' ', ''))
# print(s1)

# 方式三
l2 = ['Ra  in', 'C ats', 'and', 'Do g']
l2_1 = []
for i in l2:
    temp = i.split(' ')
    l2_1.append(''.join(temp))
print(l2_1)

# 方式四
s = ','.join(l2)
s = s.replace(' ', '')
print(s.split(','))
