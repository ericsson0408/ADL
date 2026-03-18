#!/bin/bash
echo "======================================="
echo "開始下載檔案..."
echo "======================================="
gdown --folder https://drive.google.com/drive/folders/1ByEuAoWT2-pZBMmNIPwulRvkWRL5XVPk
echo "======================================="
echo "下載完成，開始解壓縮..."
echo "======================================="
unzip HW1.zip
rm HW1.zip
echo "======================================="
echo "檔案處理完成"
echo "======================================="
