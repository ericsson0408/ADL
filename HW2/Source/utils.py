from transformers import BitsAndBytesConfig
import torch
def get_prompt(instruction: str) -> str:
    """
    接收一個指令字串，並回傳格式化後的完整提示 (Prompt)。
    Args:
        instruction (str): 來自資料集的原始指令。
    Returns:
        str: 準備輸入給模型的完整提示模板。
    """
    return f"翻譯任務。USER: {instruction} ASSISTANT:"
# def get_prompt(instruction: str) -> str:
#     """
#     接收一個指令字串，並回傳格式化後的完整提示 (Prompt)。
#     Args:
#         instruction (str): 來自資料集的原始指令。
#     Returns:
#         str: 準備輸入給模型的完整提示模板。
#     """
#     return f"你擅長文言文跟現代文的轉換。USER: {instruction} ASSISTANT:"

# def get_prompt(instruction: str) -> str:
#     """
#     接收一個指令字串，並回傳格式化後的完整提示 (Prompt)。
#     Args:
#         instruction (str): 來自資料集的原始指令。
#     Returns:
#         str: 準備輸入給模型的完整提示模板。
#     """
#     # 這就是我們在訓練腳本中使用的標準模板
#     prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
#     return prompt

def get_bnb_config() -> BitsAndBytesConfig:
    """
    建立並回傳用於 4-bit 量化的 BitsAndBytesConfig 物件。
    Returns:
        BitsAndBytesConfig: QLoRA 的設定物件。
    """
    # 這就是我們在訓練腳本中使用的 QLoRA 量化設定
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    return config
