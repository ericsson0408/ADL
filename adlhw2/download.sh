#!/bin/bash
echo "======================================="
echo "開始下載檔案..."
echo "======================================="
gdown --folder https://drive.google.com/drive/folders/1Z2A3rKplrkWPql2JSYxaqI6Pe-cLTQOc
echo "======================================="
echo "下載完成，開始解壓縮..."
echo "======================================="
unzip adapter_checkpoint.zip
rm adapter_checkpoint.zip
echo "======================================="
echo "檔案處理完成"
echo "======================================="

