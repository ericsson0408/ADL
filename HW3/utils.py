from typing import List
import re
    
def get_inference_system_prompt() -> str:
    """get system prompt for generation"""
    prompt = (
        "You are a helpful and precise question-answering assistant. "
        "You must use ONLY the provided context to answer the user's query. "
        "If the answer is not contained within the context, respond with 'CANNOTANSWER'"
        )
    return prompt

def get_inference_user_prompt(query : str, context_list : List[str]) -> str:
    context_str = "\n".join([f"Context {i+1}: {c}" for i, c in enumerate(context_list)])
    prompt = f"""Here are the relevant passages retrieved from the knowledge base:
        --- CONTEXT ---
        {context_str}
        --- END CONTEXT ---
        Based strictly on the context provided above, please answer the following query.
        --- QUERY ---
        {query}
        --- END QUERY ---
        Answer:"""
    return prompt
def parse_generated_answer(pred_ans: str) -> str:
    """Extract the actual answer from the model's generated text."""
    if "Answer:" in pred_ans:
        parsed_ans = pred_ans.split("Answer:")[-1]
    else:
        parsed_ans = pred_ans
    parsed_ans = parsed_ans.strip()
    unwanted_prefix = "assistant\n<think>\n\n</think>\n\n"
    if parsed_ans.startswith(unwanted_prefix):
        parsed_ans = parsed_ans[len(unwanted_prefix):]
    return parsed_ans.strip()