m = int(input('请输入你的资金'))
if m >= 100:
    print('可以购买一等座')
elif 0 < m < 100:
    print('可以买二等座')
else:
    print('请输入大于0的金额')