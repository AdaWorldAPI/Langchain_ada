# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain

# 🔄 Load .env variables
load_dotenv()

# 🔐 Fetch API keys
gemini_key = os.getenv("GOOGLE_API_KEY")

# ✅ Check: print key partially (safe check)
print("Using Gemini key:", gemini_key[:4], "..." if gemini_key else "❌ MISSING")

# ✅ Use correct variable
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=gemini_key  # <-- FIXED
)

# ⚠️ Deprecated (can upgrade later)
chain = ConversationChain(llm=llm, verbose=True)

try:
    response = chain.invoke("Hello Ada, what's your purpose?")
    print("💬 Ada:", response)
except Exception as e:
    print("❌ LangChain Error:", e)
