import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
vs = client.vector_stores.create(name="PastPulse Faculty Notes")
print(vs.id)
