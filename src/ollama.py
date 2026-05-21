import json
from openai import OpenAI
import random
import concurrent.futures
from time import sleep

# List of base URLs for the OpenAI
base_urls = ["http://localhost:11434/v1"]

def ollama(user_message, system_prompt, model="gpt4o", history=None, max_tokens=256, temperature=0, top_p=1.0, frequency_penalty=0, presence_penalty=0, retry=5, index=-1):
    if model == "llama3.1":
        model_name = "llama3.1"
    elif model == "phi3":
        model_name = "phi3:3.8b-mini-4k-instruct-fp16"
    elif model == "qwen2.5":
        model_name = "qwen2.5"
    elif model == "phi3.5":
        model_name = "phi3.5"
    elif model == "gemma2":
        model_name = "gemma2"
    elif model == "mistralv3":
        model_name = "mistral"
    elif model == "gpt-oss:20b":
        model_name = "gpt-oss:20b"
    elif model == "gpt-oss:120b":
        model_name = "gpt-oss:120b"
    elif model == "qwen3:4b":
        model_name = "qwen3:4b"
    else:
        raise ValueError("Invalid model name. Please provide a valid model name.")

    # If base_url is not provided, then select a random base_url from the list
    if len(base_urls) <= 1:
        base_url = base_urls[0]
    elif index >= 0 and index < len(base_urls):
        base_url = base_urls[index]
    else:
        base_url = random.choice(base_urls)
     
     # If history is None, then it is the first message of the conversation
    messages = [{"role": "system", "content": system_prompt}]
    if history is not None:
        for turn in history:
            messages.append({"role": "user", "content": turn['User']})
            messages.append({"role": "assistant", "content": turn['Assistant']})
    messages += [{"role": "user", "content": user_message}]

    flag = True
    original_retry = retry  # Store original retry count for calculating attempt number
    while(flag):
        try:
            client = OpenAI(base_url=base_url, api_key="ollama")
            response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=False)

            gen_response = response.choices[0].message.content
            token_details = response.usage
            flag = False
        except Exception as e:
            attempt = original_retry - retry  # Calculate current attempt number
            print(f"Ollama error (attempt {attempt + 1}/{original_retry}): {e}")
            gen_response = "Could not get a response"
            token_details = {}
            token_details["prompt_tokens"] = 0
            token_details["completion_tokens"] = 0
            token_details["total_tokens"] = 0
            
            if retry > 0:
                retry -= 1
                if retry > 0:  # Don't sleep on the last attempt
                    wait_time = (2 ** attempt) + (attempt * 0.1)  # Exponential backoff with jitter
                    print(f"Retrying in {wait_time:.1f} seconds...")
                    sleep(wait_time)
            else:
                flag = False

    if gen_response == "Could not get a response":
        print("Retry Count exceeded. Could not get a response. Please check the API or your code.")
    
    return user_message, gen_response, token_details

# batch inference of ollama where prompts is a list of user messages and that will be uniformly distributed among the base_urls
def ollama_batch_inference(prompts, system_prompt, model="gpt4o", max_tokens=256, temperature=0, top_p=1.0, frequency_penalty=0, presence_penalty=0, retry=5):
    response_dict = {}
    # Split prompts into equal parts for each base_url
    num_prompts = len(prompts)
    num_base_urls = len(base_urls)
    prompts_per_base_url = num_prompts // num_base_urls
    # but last base_url will have the remaining prompts
    for i in range(num_base_urls):
        start = i * prompts_per_base_url
        end = (i+1) * prompts_per_base_url
        if i == num_base_urls-1:
            end = num_prompts
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit threads
            futures = []
            for prompt in prompts[start:end]:
                futures.append(executor.submit(ollama, user_message=prompt, system_prompt=system_prompt, model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, retry=retry, index=i))

            # Extract output from completed threads
            for future in concurrent.futures.as_completed(futures):
                output = future.result()
                response_dict[output[0]] = output[1]

    # Order the responses based on the order of prompts
    responses = []
    for prompt in prompts:
        if prompt in response_dict:
            responses.append(response_dict[prompt])
        else:
            responses.append("Could not get a response")
    return responses