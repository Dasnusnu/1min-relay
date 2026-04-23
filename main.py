from flask import Flask, request, jsonify, make_response, Response
import requests
import time
import uuid
import warnings
from waitress import serve
import json
import tiktoken
import socket
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from pymemcache.client.base import Client
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import logging
from io import BytesIO
import coloredlogs
import printedcolors
import base64

# Suppress warnings from flask_limiter
warnings.filterwarnings("ignore", category=UserWarning, module="flask_limiter.extension")

# Create a logger object
logger = logging.getLogger("1min-relay")

# Install coloredlogs with desired log level
coloredlogs.install(level='DEBUG', logger=logger)

def check_memcached_connection(host='memcached', port=11211):
    try:
        client = Client((host, port))
        client.set('test_key', 'test_value')
        if client.get('test_key') == b'test_value':
            client.delete('test_key')  # Clean up
            return True
        else:
            return False
    except:
        return False
        
logger.info('''
    _ __  __ _      ___     _           
 / |  \/  (_)_ _ | _ \___| |__ _ _  _ 
 | | |\/| | | ' \|   / -_) / _` | || |
 |_|_|  |_|_|_||_|_|_\___|_\__,_|\_, |
                                 |__/ ''')


def calculate_token(sentence, model="DEFAULT"):
    """Calculate the number of tokens in a sentence based on the specified model."""
    
    if model.startswith("mistral"):
        # Initialize the Mistral tokenizer
        tokenizer = MistralTokenizer.v3(is_tekken=True)
        model_name = "open-mistral-nemo" # Default to Mistral Nemo
        tokenizer = MistralTokenizer.from_model(model_name)
        tokenized = tokenizer.encode_chat_completion(
            ChatCompletionRequest(
                messages=[
                    UserMessage(content=sentence),
                ],
                model=model_name,
            )
        )
        tokens = tokenized.tokens
        return len(tokens)

    elif model in ["gpt-3.5-turbo", "gpt-4"]:
        # Use OpenAI's tiktoken for GPT models
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(sentence)
        return len(tokens)

    else:
        # Default to openai
        encoding = tiktoken.encoding_for_model("gpt-4")
        tokens = encoding.encode(sentence)
        return len(tokens)
app = Flask(__name__)
if check_memcached_connection():
    limiter = Limiter(
        get_remote_address,
        app=app,
        storage_uri="memcached://memcached:11211",  # Connect to Memcached created with docker
    )
else:
    # Used for ratelimiting without memcached
    limiter = Limiter(
        get_remote_address,
        app=app,
    )
    logger.warning("Memcached is not available. Using in-memory storage for rate limiting. Not-Recommended")


ONE_MIN_API_URL = "https://api.1min.ai/api/features"
ONE_MIN_CONVERSATION_API_URL = "https://api.1min.ai/api/conversations"
ONE_MIN_CONVERSATION_API_STREAMING_URL = "https://api.1min.ai/api/features?isStreaming=true"
ONE_MIN_ASSET_URL = "https://api.1min.ai/api/assets"

# Define the models that are available for use
ALL_ONE_MIN_AVAILABLE_MODELS = [
    # OpenAI
    "gpt-4o", "gpt-4o-mini", "o1", "o1-preview", "o1-mini", "o3-mini",
    "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
    
    # Anthropic
    "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
    
    # DeepSeek
    "deepseek-chat", "deepseek-reasoner", "deepseek-v3",
    
    # Google
    "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro",
    
    # Mistral
    "mistral-large-latest", "mistral-small-latest", "mistral-nemo", "open-mistral-7b", "pixtral-12b-2409",
    
    # Meta
    "meta/meta-llama-3.1-405b-instruct", "meta/meta-llama-3.1-70b-instruct", "meta/meta-llama-3.1-8b-instruct",
    "meta/llama-2-70b-chat",

    # Cohere
    "command-r-plus", "command-r",
    
    # xAI
    "grok-1", "grok-beta"
]

# Define the models that support vision inputs
vision_supported_models = [
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "claude-3-7-sonnet-20250219", 
    "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620", "gemini-1.5-pro", 
    "gemini-1.5-flash", "pixtral-12b-2409"
]

# Define the models that support image generation
image_generation_models = [
    "stable-image", "stable-diffusion-xl-1024-v1-0", "stable-diffusion-v1-6",
    "midjourney", "dall-e-3", "black-forest-labs/flux-schnell", "black-forest-labs/flux-pro",
    "recraft-v3"
]


# Default values
SUBSET_OF_ONE_MIN_PERMITTED_MODELS = ["mistral-nemo", "gpt-4o", "deepseek-chat"]
PERMIT_MODELS_FROM_SUBSET_ONLY = False

# Read environment variables
ONE_MIN_API_KEY = os.getenv("ONE_MIN_API_KEY")
one_min_models_env = os.getenv("SUBSET_OF_ONE_MIN_PERMITTED_MODELS")
permit_not_in_available_env = os.getenv("PERMIT_MODELS_FROM_SUBSET_ONLY")

if one_min_models_env:
    SUBSET_OF_ONE_MIN_PERMITTED_MODELS = one_min_models_env.split(",")

if permit_not_in_available_env and permit_not_in_available_env.lower() == "true":
    PERMIT_MODELS_FROM_SUBSET_ONLY = True

AVAILABLE_MODELS = []
AVAILABLE_MODELS.extend(SUBSET_OF_ONE_MIN_PERMITTED_MODELS)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        internal_ip = socket.gethostbyname(socket.gethostname())
        return "Congratulations! Your API is working! You can now make requests to the API.\n\nEndpoint: " + internal_ip + ':5001/v1'
    return ERROR_HANDLER(1405)

@app.route('/v1/models')
@limiter.limit("500 per minute")
def models():
    current_models = fetch_models_from_docs()
    
    models_data = []
    if not PERMIT_MODELS_FROM_SUBSET_ONLY:
        one_min_models_data = [
            {"id": model_name, "object": "model", "owned_by": "1minai", "created": 1727389042}
            for model_name in current_models
        ]
    else:
        one_min_models_data = [
            {"id": model_name, "object": "model", "owned_by": "1minai", "created": 1727389042}
            for model_name in SUBSET_OF_ONE_MIN_PERMITTED_MODELS
        ]
    models_data.extend(one_min_models_data)
    return jsonify({"data": models_data, "object": "list"})

def ERROR_HANDLER(code, model=None, key=None):
    error_codes = {
        1002: {"message": f"The model {model} does not exist.", "type": "invalid_request_error", "param": None, "code": "model_not_found", "http_code": 400},
        1020: {"message": f"Incorrect API key provided: {key}. You can find your API key at https://app.1min.ai/api.", "type": "authentication_error", "param": None, "code": "invalid_api_key", "http_code": 401},
        1021: {"message": "Invalid Authentication. Provide a Bearer token or set ONE_MIN_API_KEY environment variable.", "type": "invalid_request_error", "param": None, "code": None, "http_code": 401},
        1212: {"message": f"Incorrect Endpoint. Please use the /v1/chat/completions endpoint.", "type": "invalid_request_error", "param": None, "code": "model_not_supported", "http_code": 400},
        1044: {"message": f"This model does not support image inputs.", "type": "invalid_request_error", "param": None, "code": "model_not_supported", "http_code": 400},
        1412: {"message": f"No message provided.", "type": "invalid_request_error", "param": "messages", "code": "invalid_request_error", "http_code": 400},
        1423: {"message": f"No content in last message.", "type": "invalid_request_error", "param": "messages", "code": "invalid_request_error", "http_code": 400},
        1405: {"message": "Method Not Allowed", "type": "invalid_request_error", "param": None, "code": None, "http_code": 405},
    }
    error_data = {k: v for k, v in error_codes.get(code, {"message": "Unknown error", "type": "unknown_error", "param": None, "code": None}).items() if k != "http_code"}
    logger.error(f"An error has occurred while processing the user's request. Error code: {code}")
    return jsonify({"error": error_data}), error_codes.get(code, {}).get("http_code", 400)

def format_conversation_history(messages):
    """
    Formats the conversation history into a structured string.
    """
    formatted_history = []
    
    for message in messages:
        role = message.get('role', '').capitalize()
        content = message.get('content', '')
        
        if isinstance(content, list):
            text_parts = [item['text'] for item in content if isinstance(item, dict) and item.get('type') == 'text']
            content = '\n'.join(text_parts)
        
        formatted_history.append(f"{role}: {content}")
    
    return '\n'.join(formatted_history)


@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def conversation():
    if request.method == 'OPTIONS':
        return handle_options_request()

    auth_header = request.headers.get('Authorization')
    api_key = ONE_MIN_API_KEY
    if auth_header and auth_header.startswith("Bearer "):
        api_key = auth_header.split(" ")[1]
        
    if not api_key:
        logger.error("Invalid Authentication")
        return ERROR_HANDLER(1021)
    
    headers = {'API-KEY': api_key, 'Content-Type': 'application/json'}
    
    request_data = request.json
    messages = request_data.get('messages', [])
    if not messages:
        return ERROR_HANDLER(1412)

    model = request_data.get('model', 'mistral-nemo')
    all_messages_str = format_conversation_history(messages)
    
    # Process images if present in the last message (OpenAI format)
    image_paths = []
    has_image = False
    last_message = messages[-1]
    
    if isinstance(last_message.get('content'), list):
        if model not in vision_supported_models:
            return ERROR_HANDLER(1044, model)
            
        for item in last_message['content']:
            if isinstance(item, dict) and item.get('type') == 'image_url':
                img_url = item['image_url']['url']
                try:
                    if img_url.startswith("data:image/"):
                        header, base64_image = img_url.split(",", 1)
                        mime_type = header.split(";")[0].split(":")[1]
                        binary_data = base64.b64decode(base64_image)
                    else:
                        img_response = requests.get(img_url)
                        img_response.raise_for_status()
                        binary_data = img_response.content
                        mime_type = img_response.headers.get('Content-Type', 'image/png')
                    
                    files = {'asset': (f"relay-{uuid.uuid4()}", binary_data, mime_type)}
                    asset_resp = requests.post(ONE_MIN_ASSET_URL, files=files, headers={'API-KEY': api_key})
                    asset_resp.raise_for_status()
                    image_paths.append(asset_resp.json()['fileContent']['path'])
                    has_image = True
                except Exception as e:
                    logger.error(f"Error processing image: {e}")

    prompt_token = calculate_token(all_messages_str)
    
    if PERMIT_MODELS_FROM_SUBSET_ONLY and model not in AVAILABLE_MODELS:
        return ERROR_HANDLER(1002, model)
    
    logger.debug(f"Processing {prompt_token} tokens with model {model}")

    if not has_image:
        payload = {
            "type": "CHAT_WITH_AI",
            "model": model,
            "promptObject": {"prompt": all_messages_str, "isMixed": False, "webSearch": False}
        }
    else:
        payload = {
            "type": "CHAT_WITH_IMAGE",
            "model": model,
            "promptObject": {"prompt": all_messages_str, "isMixed": False, "imageList": image_paths}
        }
    
    if not request_data.get('stream', False):
        response = requests.post(ONE_MIN_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        one_min_response = response.json()
        
        transformed_response = transform_response(one_min_response, request_data, prompt_token)
        response = make_response(jsonify(transformed_response))
        set_response_headers(response)
        return response, 200
    else:
        response_stream = requests.post(ONE_MIN_CONVERSATION_API_STREAMING_URL, json=payload, headers=headers, stream=True)
        if response_stream.status_code != 200:
            return ERROR_HANDLER(response_stream.status_code)
        return Response(stream_response(response_stream, request_data, model, int(prompt_token)), content_type='text/event-stream')


@app.route('/v1/images/generations', methods=['POST', 'OPTIONS'])
@limiter.limit("100 per minute")
def generate_images():
    if request.method == 'OPTIONS':
        return handle_options_request()

    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error("Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]
    headers = {
        'API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    request_data = request.json
    prompt = request_data.get('prompt')
    if not prompt:
        return ERROR_HANDLER(1412)  # No prompt provided

    num_images = request_data.get('n', 1)
    image_size = request_data.get('size', "1024x1024")
    model = request_data.get('model', 'black-forest-labs/flux-schnell')
    if not model in image_generation_models:
        return ERROR_HANDLER(1044, model)

    payload = {
        "type": "IMAGE_GENERATOR",
        "model": model,
        "promptObject": {
            "prompt": prompt,
            "n": num_images,
            "size": image_size
        }
    }

    try:
        logger.debug("Image Generation Requested.")
        response = requests.post(ONE_MIN_API_URL + "?isStreaming=false", json=payload, headers=headers)
        response.raise_for_status()
        one_min_response = response.json()

        image_urls = one_min_response['aiRecord']['aiRecordDetail']['resultObject']
        transformed_response = {
            "created": int(time.time()),
            "data": [{"url": url} for url in image_urls]
        }

        return jsonify(transformed_response), 200

    except requests.exceptions.RequestException as e:
        logger.error(f"Image generation failed: {str(e)}")
        return ERROR_HANDLER(1044)  # Handle image generation error

def handle_options_request():
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response, 204

def transform_response(one_min_response, request_data, prompt_token):
    completion_token = calculate_token(one_min_response['aiRecord']["aiRecordDetail"]["resultObject"][0])
    logger.debug(f"Finished processing Non-Streaming response. Completion tokens: {str(completion_token)}")
    logger.debug(f"Total tokens: {str(completion_token + prompt_token)}")
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request_data.get('model', 'mistral-nemo'),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": one_min_response['aiRecord']["aiRecordDetail"]["resultObject"][0],
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_token,
            "completion_tokens": completion_token,
            "total_tokens": prompt_token + completion_token
        }
    }
    
def set_response_headers(response):
    response.headers['Content-Type'] = 'application/json'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['X-Request-ID'] = str (uuid.uuid4())

def stream_response(response, request_data, model, prompt_tokens):
    all_chunks = ""
    for chunk in response.iter_content(chunk_size=1024):
        finish_reason = None

        return_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request_data.get('model', 'mistral-nemo'),
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": chunk.decode('utf-8')
                    },
                    "finish_reason": finish_reason
                }
            ]
        }
        all_chunks += chunk.decode('utf-8')
        yield f"data: {json.dumps(return_chunk)}\n\n"
        
    tokens = calculate_token(all_chunks)
    logger.debug(f"Finished processing streaming response. Completion tokens: {str(tokens)}")
    logger.debug(f"Total tokens: {str(tokens + prompt_tokens)}")
        
    # Final chunk when iteration stops
    final_chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request_data.get('model', 'mistral-nemo'),
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": ""    
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": tokens,
            "total_tokens": tokens + prompt_tokens
        }
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

if __name__ == '__main__':
    internal_ip = socket.gethostbyname(socket.gethostname())
    response = requests.get('https://api.ipify.org')
    public_ip = response.text
    logger.info(f"""{printedcolors.Color.fg.lightcyan}  
Server is ready to serve at:
Internal IP: {internal_ip}:5001
Public IP: {public_ip} (only if you've setup port forwarding on your router.)
Enter this url to OpenAI clients supporting custom endpoint:
{internal_ip}:5001/v1
If does not work, try:
{internal_ip}:5001/v1/chat/completions
{printedcolors.Color.reset}""")
    serve(app, host='0.0.0.0', port=5001, threads=6) # Thread has a default of 4 if not specified. We use 6 to increase performance and allow multiple requests at once.

al tokens: {str(tokens + prompt_tokens)}")
        
    # Final chunk when iteration stops
    final_chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request_data.get('model', 'mistral-nemo'),
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": ""    
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": tokens,
            "total_tokens": tokens + prompt_tokens
        }
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

if __name__ == '__main__':
    internal_ip = socket.gethostbyname(socket.gethostname())
    response = requests.get('https://api.ipify.org')
    public_ip = response.text
    logger.info(f"""{printedcolors.Color.fg.lightcyan}  
Server is ready to serve at:
Internal IP: {internal_ip}:5001
Public IP: {public_ip} (only if you've setup port forwarding on your router.)
Enter this url to OpenAI clients supporting custom endpoint:
{internal_ip}:5001/v1
If does not work, try:
{internal_ip}:5001/v1/chat/completions
{printedcolors.Color.reset}""")
    serve(app, host='0.0.0.0', port=5001, threads=6) # Thread has a default of 4 if not specified. We use 6 to increase performance and allow multiple requests at once.

