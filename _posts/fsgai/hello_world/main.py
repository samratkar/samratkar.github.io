from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        { "role": "user", "content": "Hey, I am Piyush Garg! Nice to meet you"}
    ]
)

print(response.choices[0].message.content)