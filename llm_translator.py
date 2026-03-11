import requests
import json

def stream_translated_report(report: str, selected_language: str):
    """
    Sends the generated medical report to DeepSeek for cleaning and translation.
    Yields chunks of the final text as they arrive.
    """
    prompt = f"""You are a medical report editor.

Your task:
1. Clean and refine the following medical report.
2. Correct grammar and formatting.
3. Preserve all medical terminology and values.
4. Translate the final cleaned report into {selected_language}.
5. Output only the final formatted report.

Report:
{report}"""

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    # Updated payload to match your requested DeepSeek V3.2 settings
    payload = {
        "model": "deepseek-ai/deepseek-v3.2",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
        "stream": True,
        "extra_body": {"chat_template_kwargs": {"thinking": False}}
    }
    
    response = requests.post(url, headers=headers, json=payload, stream=True)
    
    # If there is an authentication or bad request error, raise it
    response.raise_for_status()
    
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: "):
                data_str = decoded_line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        
                        c_chunk = delta.get("content", "")
                        if c_chunk:
                            yield c_chunk
                except json.JSONDecodeError:
                    pass
