import subprocess
import re
import matplotlib.pyplot as plt
import pandas as pd
import sys

def main():
    """
    完全模擬 runrun.sh 的行為，執行 ppl.py 並自動捕捉結果繪圖。
    不需對您原本的 ppl.py 做任何修改。
    """
    checkpoints = []
    perplexities = []
    
    # ------------------- 您需要確認的參數 -------------------
    # 請確認 python 直譯器的名稱是 python3 還是 python
    PYTHON_EXECUTABLE = "python3" 
    # 請確認您的 peft adapter 資料夾路徑
    PEFT_DIR = "adapter_shortprompt"
    # 請確認您的測試資料路徑
    TEST_DATA_PATH = "public_test.json"
    # 請確認您的基礎模型路徑
    BASE_MODEL_PATH = "Qwen/Qwen3-4B"
    # ----------------------------------------------------

    print("===== 開始執行自動化評估腳本 =====")

    for i in range(1, 13):
        peft_value = 100 * i
        peft_path = f"{PEFT_DIR}/checkpoint-{peft_value}"
        
        # 組成要執行的命令
        command = [
            PYTHON_EXECUTABLE, "ppl.py",
            "--base_model_path", BASE_MODEL_PATH,
            "--peft_path", peft_path,
            "--test_data_path", TEST_DATA_PATH
        ]
        
        print(f"\n[ {i}/12 ] 正在執行命令:")
        print(" ".join(command)) # 將指令完整印出，方便檢查

        try:
            # 執行指令，如果出錯會拋出異常
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,        # 將輸出解碼為文字
                check=True,       # 如果回傳非 0 (錯誤)，就拋出 CalledProcessError
                encoding='utf-8'  # 明確指定編碼
            )
            
            # 從標準輸出中尋找 perplexity
            output = result.stdout
            print("--- 來自 ppl.py 的輸出 ---")
            print(output.strip())
            print("--------------------------")

            match = re.search(r"Mean perplexity:\s*([\d.]+)", output)
            
            if match:
                perplexity = float(match.group(1))
                checkpoints.append(peft_value)
                perplexities.append(perplexity)
                print(f"✅ 成功捕捉到 Checkpoint {peft_value} 的結果: {perplexity}")
            else:
                print(f"⚠️ 警告: 在 Checkpoint {peft_value} 的輸出中找不到 'Mean perplexity'。")

        except FileNotFoundError:
            print(f"致命錯誤: 找不到 '{PYTHON_EXECUTABLE}'。請確認您的 Python 環境是否設定正確。")
            sys.exit(1) # 終止程式
        except subprocess.CalledProcessError as e:
            # 如果 ppl.py 執行失敗，這會捕捉到錯誤並印出
            print(f"❌ 錯誤: 執行 Checkpoint {peft_value} 時，ppl.py 腳本出錯。")
            print("--- 錯誤訊息 (stderr) ---")
            print(e.stderr.strip())
            print("--------------------------")
            print("由於發生錯誤，腳本將會終止。")
            sys.exit(1) # 終止程式

    if not checkpoints:
        print("\n評估未完成或未捕捉到任何數據，無法生成圖表。")
        return

    print("\n===== 評估完成，開始繪製圖表 =====")
    
    # --- 繪圖區塊 (與之前相同) ---
    df = pd.DataFrame({'checkpoint': checkpoints, 'mean_perplexity': perplexities})
    plt.figure(figsize=(12, 7))
    plt.plot(df['checkpoint'], df['mean_perplexity'], marker='o', linestyle='-')
    for idx, row in df.iterrows():
        plt.text(row['checkpoint'], row['mean_perplexity'], f"{row['mean_perplexity']:.2f}", ha='center', va='bottom')
    plt.title('Mean Perplexity vs. Checkpoint', fontsize=16)
    plt.xlabel('Checkpoint Step', fontsize=12)
    plt.ylabel('Mean Perplexity', fontsize=12)
    plt.xticks(checkpoints, rotation=45)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    output_filename = "perplexity_chart_final.png"
    plt.savefig(output_filename)
    print(f"✅ 圖表已儲存為 '{output_filename}'")

if __name__ == "__main__":
    main()