from llama_cloud_services import LlamaParse
import os
from dotenv import load_dotenv

load_dotenv(verbose=True)
parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    num_workers=2,
    verbose=True,
    language="en",
)

result = parser.parse(file_path="tr_technology_radar_vol_33_cn.pdf")

markdown_documents = result.get_markdown_documents(split_by_page=True)
text_documents = result.get_text_documents(split_by_page=False)

image_documents = result.get_image_documents(
    include_screenshot_images=True,
    include_object_images=False,
    image_download_dir="./images",
)

for page in result.pages:
    print(page.text)
    print(page.md)
    print(page.images)
    print(page.layout)
    print(page.structuredData)