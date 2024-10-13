from dotenv import load_dotenv
import warnings
from pathlib import Path
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from PIL import Image
import matplotlib.pyplot as plt
import os
from llama_index.core import SimpleDirectoryReader

warnings.filterwarnings('ignore')
_ = load_dotenv()

# Load Images with urls

image_urls = [
    "https://res.cloudinary.com/hello-tickets/image/upload/c_limit,f_auto,q_auto,w_1920/v1640835927/o3pfl41q7m5bj8jardk0.jpg",
]

image_documents = load_image_urls(image_urls)

openai_mm_llm = OpenAIMultiModal(
    model="gpt-4o-mini", max_new_tokens=300
)

response = openai_mm_llm.complete(
    prompt="Describe the images as an alternative text",
    image_documents=image_documents,
)

print(response)

# Load Images from directory

input_image_path = Path("input_images")
if not input_image_path.exists():
    Path.mkdir(input_image_path)

def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 9:
                break

image_paths = []
for img_path in os.listdir("../input_images"):
    image_paths.append(str(os.path.join("../input_images", img_path)))
plot_images(image_paths)

# put your local directory here
image_documents = SimpleDirectoryReader("../input_images").load_data()

response = openai_mm_llm.complete(
    prompt="Describe the images as an alternative text",
    image_documents=image_documents,
)

print(response)
