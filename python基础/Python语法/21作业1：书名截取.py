bookList = ['Java程序设计', '西游记', '鲁滨孙漂流记', '我的独孤是一座花园', '植物的记忆与藏书乐', '围城',
            '爱丽丝漫游奇境', '年轻不老，老得年轻', '文化遗产的保护与传播', '三国演义']
priList = []
for i in bookList:
    if len(i) > 8:
        priList.append(i[:8]+'...')
        # print(i[:8]+'...')
    else:
        priList.append(i)
# print(priList)
print(','.join(priList))