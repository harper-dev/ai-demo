import os
import requests
from pathlib import Path
from dotenv import load_dotenv
import asyncio
from llama_cloud_services import LlamaParse

load_dotenv(verbose=True)
sample_dir = Path("./data/pdfs")
sample_dir.mkdir(parents=True, exist_ok=True)

sample_docs = {
    "attention.pdf": "https://arxiv.org/pdf/1706.03762.pdf",
    "bert.pdf": "https://arxiv.org/pdf/1810.04805.pdf",
}

for file_name, url in sample_docs.items():
    file_path = sample_dir / file_name
    if not file_path.exists():
        print(f"Downloading {file_name}...")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            #Basic content validation
            if response.headers.get("Content-Type", "").startswith("application/pdf"):
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded {file_name} successfully.")
            else:
                print(f"Failed to download {file_name}: Invalid pdf.")
        except requests.RequestException as e:
            print(f"Failed to download {file_name}: {e}")
    else:
        print(f"{file_name} already exists, skipping download.")
print("\n Sample files ready!")

pdf_files = list(sample_dir.glob("*.pdf"))

parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    num_workers=1,
    show_progress=False,
    verbose=True,
)

semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent tasks

async def parse_single_file(
    parser,
    file_path,
    semaphore,
):
    async with semaphore:
        try:
            print(f"Starting parse for {file_path.name}")
            result = await parser.aparse(str(file_path))
            print(f"Completed {file_path.name}: {len(result.pages)} pages parsed.")
            return {
                "file": file_path.name,
                "status": "success",
                "result": result,
                "pages": len(result.pages) if result else 0,
            }
        except Exception as e:
            print(f"Error parsing {file_path.name}: {e}")
            return {
                "file": file_path.name,
                "status": "error",
                "error": str(e),
            }

task = [
    parse_single_file(
        parser,
        file_path,
        semaphore,
    )
    for file_path in pdf_files
]
async def main():
    return await asyncio.gather(*task)
# results = asyncio.run(asyncio.gather(*task))
# results = asyncio.gather(*task)
results = asyncio.run(main())
# for res in results:
#     print(res)