import os
from dotenv import load_dotenv
from getpass import getpass
from llama_cloud_services import LlamaParse
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
import asyncio
load_dotenv(override=True)
async def main():
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        model="openai-gpt-4o-mini",
        high_res_ocr=True,
        adaptive_table_extraction=True,
        output_table_as_HTML=True,
        show_progress=True,
        verbose=True,
    )

    result = await parser.aparse("./data/Eqvista_DCF-Excel-Template.xlsx");
    llama_parse_documents = result.get_text_documents(split_by_page=True)
    print(llama_parse_documents[0].text)

    llm = OpenAI(model="gpt-4o-mini", temperature=0)

    query_str = "Tell me about the income taxes in the past years (year 3-5) for the 5 year WMA table"
    context = "\n\n".join([doc.text for doc in llama_parse_documents])
    messages = [
        ChatMessage(
            role="user",
            content=f"Here is some context\n<context>{context}</context>\n\nAnswer the following question: {query_str}",
        )
    ]

    response = await llm.achat(messages)
    print(response.message.content)

if __name__ == "__main__":
    asyncio.run(main())
