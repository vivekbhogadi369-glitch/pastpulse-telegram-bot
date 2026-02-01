import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Correct for OpenAI SDK v2.x
vs = client.beta.vector_stores.create(
    name="PastPulse Faculty Notes"
)

print("VECTOR_STORE_ID =", vs.id)
