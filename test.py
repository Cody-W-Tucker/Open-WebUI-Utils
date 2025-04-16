from pipelines.agent import Pipeline
import asyncio


async def main():
    pipeline = Pipeline()
    await pipeline.on_startup()
    
    # Create a list of message dictionaries to represent chat history
    messages = []  # Empty for first message, would contain previous exchanges for ongoing conversation
    
    response = pipeline.pipe(
        user_message="What is the knowledge crisis and how can we solve it?",
        messages=messages
    )
    
    print(response)
    await pipeline.on_shutdown()

if __name__ == "__main__":
    asyncio.run(main())
