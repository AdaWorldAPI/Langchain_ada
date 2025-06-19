# test_direct.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain

gemini_key = "AIzaSyCjTTGlJVzCH5z6KfDh686ihwqZTIUt63k"  # 🔐 insert your working key directly here

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=gemini_key
)

chain = ConversationChain(llm=llm, verbose=True)
response = chain.run("Hello Ada, what's your purpose?")
print(response)
