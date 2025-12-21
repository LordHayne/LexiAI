import asyncio

async def call_model_async(chat_client, messages_or_prompt):
    if isinstance(messages_or_prompt, list):
        prompt = "\n".join([m["content"] for m in messages_or_prompt])
        if hasattr(chat_client, "ainvoke"):
            return await chat_client.ainvoke(prompt)
        return chat_client.invoke(prompt)
    return chat_client.invoke(messages_or_prompt)

def call_model(chat_client, messages_or_prompt):
    if asyncio.iscoroutinefunction(chat_client.invoke) or hasattr(chat_client, "ainvoke"):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(call_model_async(chat_client, messages_or_prompt))
    return chat_client.invoke(messages_or_prompt)
