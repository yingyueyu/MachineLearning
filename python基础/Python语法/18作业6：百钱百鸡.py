# g 为公鸡5元一只   m 为母鸡3元一只  x 为小鸡1元三只
for g in range(100 // 5):
    for m in range(100 // 3):
        for x in range(100):
            if g + m + x * 3 == 100 and g * 5 + m * 3 + x == 100:
                print(f'公鸡{g}只，母鸡{m}只，小鸡{x * 3}只')