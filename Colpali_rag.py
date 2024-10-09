from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import MllamaForConditionalGeneration
from transformers import AutoModelForCausalLM
from groq import Groq
from docx2pdf import convert
from byaldi import RAGMultiModalModel
from models.converters import convert_docs_to_pdfs
from io import BytesIO
import hashlib
from models.model_loader import load_model
from transformers import GenerationConfig
import google.generativeai as genai
from openai import OpenAI
from PIL import Image
import torch
import base64
import os
from logger import get_logger
import logging


from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


logger = get_logger(__name__)

# Cache for loaded models
_model_cache = {}


# logger.py
def get_logger(name):
    """
    Creates a logger with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Console handler
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)

        # File handler
        f_handler = logging.FileHandler('app.log')
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add them to handlers
        c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger

# models/model_loader.py
def detect_device():
    """
    Detects the best available device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def load_model(model_choice):
    """
    Loads and caches the specified model.
    """
    global _model_cache

    if model_choice in _model_cache:
        logger.info(f"Model '{model_choice}' loaded from cache.")
        return _model_cache[model_choice]

    if model_choice == 'qwen':
        device = detect_device()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        model.to(device)
        _model_cache[model_choice] = (model, processor, device)
        logger.info("Qwen model loaded and cached.")
        return _model_cache[model_choice]

    elif model_choice == 'gemini':
        # Load Gemini model
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-002')  # Use the appropriate model name
        return model, None


    elif model_choice == 'llama-vision':
        # Load Llama-Vision model
        device = detect_device()
        # model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        model_id = "alpindale/Llama-3.2-11B-Vision-Instruct"
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
        model.to(device)
        _model_cache[model_choice] = (model, processor, device)
        logger.info("Llama-Vision model loaded and cached.")
        return _model_cache[model_choice]

    elif model_choice == "pixtral":
        device = detect_device()
        mistral_models_path = os.path.join(os.getcwd(), 'mistral_models', 'Pixtral')

        if not os.path.exists(mistral_models_path):
            os.makedirs(mistral_models_path, exist_ok=True)
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="mistralai/Pixtral-12B-2409",
                              allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"],
                              local_dir=mistral_models_path)

        from mistral_inference.transformer import Transformer
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from mistral_common.generate import generate

        tokenizer = MistralTokenizer.from_file(os.path.join(mistral_models_path, "tekken.json"))
        model = Transformer.from_folder(mistral_models_path)

        _model_cache[model_choice] = (model, tokenizer, generate, device)
        logger.info("Pixtral model loaded and cached.")
        return _model_cache[model_choice]

    elif model_choice == "molmo":
        device = detect_device()
        processor = AutoProcessor.from_pretrained(
            'allenai/MolmoE-1B-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        model = AutoModelForCausalLM.from_pretrained(
            'allenai/MolmoE-1B-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        _model_cache[model_choice] = (model, processor, device)
        return _model_cache[model_choice]
    elif model_choice == 'groq-llama-vision':
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        client = Groq(api_key=api_key)
        _model_cache[model_choice] = client
        logger.info("Groq Llama Vision model loaded and cached.")
        return _model_cache[model_choice]
    else:
        logger.error(f"Invalid model choice: {model_choice}")
        raise ValueError("Invalid model choice.")


# models/converters.py
def convert_docs_to_pdfs(folder_path):
    """
    Converts .doc and .docx files in the folder to PDFs.

    Args:
        folder_path (str): The path to the folder containing documents.
    """
    try:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.doc', '.docx')):
                doc_path = os.path.join(folder_path, filename)
                pdf_path = os.path.splitext(doc_path)[0] + '.pdf'
                convert(doc_path, pdf_path)
                logger.info(f"Converted '{filename}' to PDF.")
    except Exception as e:
        logger.error(f"Error converting documents to PDFs: {e}")
        raise


# models/indexer.py
def index_documents(folder_path, index_name='document_index', index_path=None, indexer_model='vidore/colpali'):
    """
    Indexes documents in the specified folder using Byaldi.

    Args:
        folder_path (str): The path to the folder containing documents to index.
        index_name (str): The name of the index to create or update.
        index_path (str): The path where the index should be saved.
        indexer_model (str): The name of the indexer model to use.

    Returns:
        RAGMultiModalModel: The RAG model with the indexed documents.
    """
    try:
        logger.info(f"Starting document indexing in folder: {folder_path}")
        # Convert non-PDF documents to PDFs
        convert_docs_to_pdfs(folder_path)
        logger.info("Conversion of non-PDF documents to PDFs completed.")

        # Initialize RAG model
        RAG = RAGMultiModalModel.from_pretrained(indexer_model)
        if RAG is None:
            raise ValueError(f"Failed to initialize RAGMultiModalModel with model {indexer_model}")
        logger.info(f"RAG model initialized with {indexer_model}.")

        # Index the documents in the folder
        RAG.index(
            input_path=folder_path,
            index_name=index_name,
            store_collection_with_index=True,
            overwrite=True
        )

        logger.info(f"Indexing completed. Index saved at '{index_path}'.")

        return RAG
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        raise


# models/retriever.py
def retrieve_documents(RAG, query, session_id, k=3):
    """
    Retrieves relevant documents based on the user query using Byaldi.

    Args:
        RAG (RAGMultiModalModel): The RAG model with the indexed documents.
        query (str): The user's query.
        session_id (str): The session ID to store images in per-session folder.
        k (int): The number of documents to retrieve.

    Returns:
        list: A list of image filenames corresponding to the retrieved documents.
    """
    try:
        logger.info(f"Retrieving documents for query: {query}")
        results = RAG.search(query, k=k)
        images = []
        session_images_folder = os.path.join('static', 'images', session_id)
        os.makedirs(session_images_folder, exist_ok=True)

        for i, result in enumerate(results):
            if result.base64:
                image_data = base64.b64decode(result.base64)
                image = Image.open(BytesIO(image_data))

                # Generate a unique filename based on the image content
                image_hash = hashlib.md5(image_data).hexdigest()
                image_filename = f"retrieved_{image_hash}.png"
                image_path = os.path.join(session_images_folder, image_filename)

                if not os.path.exists(image_path):
                    image.save(image_path, format='PNG')
                    logger.debug(f"Retrieved and saved image: {image_path}")
                else:
                    logger.debug(f"Image already exists: {image_path}")

                # Store the relative path from the static folder
                relative_path = os.path.join('images', session_id, image_filename)
                images.append(relative_path)
                logger.info(f"Added image to list: {relative_path}")
            else:
                logger.warning(f"No base64 data for document {result.doc_id}, page {result.page_num}")

        logger.info(f"Total {len(images)} documents retrieved. Image paths: {images}")
        return images
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []


# models/responder.py
# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def generate_response(images, query, session_id, resized_height=280, resized_width=280, model_choice='qwen'):
    """
    Generates a response using the selected model based on the query and images.
    """
    try:
        logger.info(f"Generating response using model '{model_choice}'.")

        # Convert resized_height and resized_width to integers
        resized_height = int(resized_height)
        resized_width = int(resized_width)

        # Ensure images are full paths
        full_image_paths = [os.path.join('static', img) if not img.startswith('static') else img for img in images]

        # Check if any valid images exist
        valid_images = [img for img in full_image_paths if os.path.exists(img)]

        if not valid_images:
            logger.warning("No valid images found for analysis.")
            return "No images could be loaded for analysis."

        if model_choice == 'qwen':
            from qwen_vl_utils import process_vision_info
            # Load cached model
            model, processor, device = load_model('qwen')
            # Ensure dimensions are multiples of 28
            resized_height = (resized_height // 28) * 28
            resized_width = (resized_width // 28) * 28

            image_contents = []
            for image in valid_images:
                image_contents.append({
                    "type": "image",
                    "image": image,  # Use the full path
                    "resized_height": resized_height,
                    "resized_width": resized_width
                })
            messages = [
                {
                    "role": "user",
                    "content": image_contents + [{"type": "text", "text": query}],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            logger.info("Response generated using Qwen model.")
            return output_text[0]

        elif model_choice == 'gemini':
            model, _ = load_model('gemini')

            try:
                content = [query]  # Add the text query first

                for img_path in valid_images:
                    if os.path.exists(img_path):
                        try:
                            img = Image.open(img_path)
                            content.append(img)
                        except Exception as e:
                            logger.error(f"Error opening image {img_path}: {e}")
                    else:
                        logger.warning(f"Image file not found: {img_path}")

                if len(content) == 1:  # Only text, no images
                    return "No images could be loaded for analysis."

                response = model.generate_content(content)

                if response.text:
                    generated_text = response.text
                    logger.info("Response generated using Gemini model.")
                    return generated_text
                else:
                    return "The Gemini model did not generate any text response."

            except Exception as e:
                logger.error(f"Error in Gemini processing: {str(e)}", exc_info=True)
                return f"An error occurred while processing the images: {str(e)}"

        elif model_choice == 'gpt4':
            api_key = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)

            try:
                content = [{"type": "text", "text": query}]

                for img_path in valid_images:
                    logger.info(f"Processing image: {img_path}")
                    if os.path.exists(img_path):
                        base64_image = encode_image(img_path)
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        })
                    else:
                        logger.warning(f"Image file not found: {img_path}")

                if len(content) == 1:  # Only text, no images
                    return "No images could be loaded for analysis."

                response = client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    max_tokens=1024
                )

                generated_text = response.choices[0].message.content
                logger.info("Response generated using GPT-4 model.")
                return generated_text

            except Exception as e:
                logger.error(f"Error in GPT-4 processing: {str(e)}", exc_info=True)
                return f"An error occurred while processing the images: {str(e)}"

        elif model_choice == 'llama-vision':
            # Load model, processor, and device
            model, processor, device = load_model('llama-vision')

            # Process images
            # For simplicity, use the first image
            image_path = valid_images[0] if valid_images else None
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                return "No valid image found for analysis."

            # Prepare messages
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": query}
                ]}
            ]
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(image, input_text, return_tensors="pt").to(device)

            # Generate response
            output = model.generate(**inputs, max_new_tokens=512)
            response = processor.decode(output[0], skip_special_tokens=True)
            return response

        elif model_choice == "pixtral":
            model, tokenizer, generate_func, device = load_model('pixtral')

            def image_to_data_url(image_path):
                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                ext = os.path.splitext(image_path)[1][1:]  # Get the file extension
                return f"data:image/{ext};base64,{encoded_string}"

            from mistral_common.protocol.instruct.messages import UserMessage, TextChunk, ImageURLChunk
            from mistral_common.protocol.instruct.request import ChatCompletionRequest

            # Prepare the content with text and images
            content = [TextChunk(text=query)]
            for img_path in valid_images[:1]:  # Use only the first image
                content.append(ImageURLChunk(image_url=image_to_data_url(img_path)))

            completion_request = ChatCompletionRequest(messages=[UserMessage(content=content)])

            encoded = tokenizer.encode_chat_completion(completion_request)

            images = encoded.images
            tokens = encoded.tokens

            out_tokens, _ = generate_func([tokens], model, images=[images], max_tokens=256, temperature=0.35,
                                          eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
            result = tokenizer.decode(out_tokens[0])

            logger.info("Response generated using Pixtral model.")
            return result

        elif model_choice == "molmo":
            model, processor, device = load_model('molmo')
            model = model.half()  # Convert model to half precision
            pil_images = []
            for img_path in valid_images[:1]:  # Process only the first image for now
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path).convert('RGB')
                        pil_images.append(img)
                    except Exception as e:
                        logger.error(f"Error opening image {img_path}: {e}")
                else:
                    logger.warning(f"Image file not found: {img_path}")

            if not pil_images:
                return "No images could be loaded for analysis."

            try:
                # Process the images and text
                inputs = processor.process(
                    images=pil_images,
                    text=query
                )

                # Move inputs to the correct device and make a batch of size 1
                # Convert float tensors to half precision, but keep integer tensors as they are
                inputs = {k: (v.to(device).unsqueeze(0).half() if v.dtype in [torch.float32, torch.float64] else
                              v.to(device).unsqueeze(0))
                if isinstance(v, torch.Tensor) else v
                          for k, v in inputs.items()}

                # Generate output
                with torch.no_grad():  # Disable gradient calculation
                    output = model.generate_from_batch(
                        inputs,
                        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                        tokenizer=processor.tokenizer
                    )

                # Only get generated tokens; decode them to text
                generated_tokens = output[0, inputs['input_ids'].size(1):]
                generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

                return generated_text

            except Exception as e:
                logger.error(f"Error in Molmo processing: {str(e)}", exc_info=True)
                return f"An error occurred while processing the images: {str(e)}"
            finally:
                # Close the opened images to free up resources
                for img in pil_images:
                    img.close()
        elif model_choice == 'groq-llama-vision':
            client = load_model('groq-llama-vision')

            content = [{"type": "text", "text": query}]

            # Use only the first image
            if valid_images:
                img_path = valid_images[0]
                if os.path.exists(img_path):
                    base64_image = encode_image(img_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
                else:
                    logger.warning(f"Image file not found: {img_path}")

            if len(content) == 1:  # Only text, no images
                return "No images could be loaded for analysis."

            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    model="llava-v1.5-7b-4096-preview",
                )
                generated_text = chat_completion.choices[0].message.content
                logger.info("Response generated using Groq Llama Vision model.")
                return generated_text
            except Exception as e:
                logger.error(f"Error in Groq Llama Vision processing: {str(e)}", exc_info=True)
                return f"An error occurred while processing the image: {str(e)}"
        else:
            logger.error(f"Invalid model choice: {model_choice}")
            return "Invalid model selected."
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"An error occurred while generating the response: {str(e)}"