import os
import subprocess
import glob

# 获取当前目录下的所有MP3文件
mp3_files = glob.glob('*.mp3')

# 检查是否找到了MP3文件
if not mp3_files:
    print("没有找到MP3文件。")
else:
    # 构造mid3iconv命令，包含所有MP3文件
    command = ['mid3iconv', '-e', 'gbk'] + mp3_files
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # 打印命令的输出（如果有的话）
        if result.stdout:
            print(f"输出: {result.stdout.decode().strip()}")
        if result.stderr:
            print(f"错误: {result.stderr.decode().strip()}")
    except subprocess.CalledProcessError as e:
        # 如果命令返回非零退出码，subprocess.run将引发CalledProcessError异常
        print(f"执行失败: {e}")