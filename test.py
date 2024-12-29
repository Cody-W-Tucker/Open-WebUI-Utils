from pipelines.chat_history_rag import Pipeline

import asyncio

async def main():
    pipeline = Pipeline()
    await pipeline.on_startup()
    response = pipeline.pipe(
        user_message="What is the knowledge crisis and how can we solve it?",
        model_id="gpt-4o-mini",
        messages=[],
        body={}
    )
    print(response)
    await pipeline.on_shutdown()

if __name__ == "__main__":
    asyncio.run(main())