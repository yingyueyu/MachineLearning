import asyncio

num = 0


# async 定义异步函数
async def a():
    print('a')
    global num
    num = 1


# 普通的同步函数
def b():
    print('b')
    global num
    num = 2


async def main():
    # asyncio.create_task() 调用异步函数并不等待结果
    # asyncio.create_task(a())
    # await 可以让有异步函数发生阻塞，等待函数结果后再执行后续内容
    # 简单的理解就是让异步函数同步调用
    # await 必须在 async 定义的函数中使用
    await a()
    b()
    print(num)


# asyncio.run() 调用异步函数并等待结果
asyncio.run(main())
print(num)
