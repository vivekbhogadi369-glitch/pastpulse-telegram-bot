import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())

vs = client.vector_stores.create(name="PastPulse Faculty Notes")
print("VECTOR_STORE_ID =", vs.id)
