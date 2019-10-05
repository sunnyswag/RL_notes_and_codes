#!/bin/bash

echo "git auto push start ..."
echo "git pull ..."
git pull
echo "git add . ..."
git add .
echo "git commit message : $1 ..."
git commit -m $1
echo "git push ..."
git push
echo "git auto push end ..."

# .git 文件是存在本地的隐藏文件，rm 和 mv都没法改动，也不会上传到远端。
# 所以一定不要轻易删除，每个设备的.git文件都是不一样的