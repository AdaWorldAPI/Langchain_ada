from glyph_agent import GlyphAgent, FeltDTO, fake_embed
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
import os

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key= "AIzaSyCjTTGlJVzCH5z6KfDh686ihwqZTIUt63k"  # 🔐 insert your working key directly here
)

# Load GlyphAgent
agent = GlyphAgent(fake_embed)
agent.load_glyphs("glyphs.json")

def build_prompt(user_input: str) -> str:
    hydrated = agent.hydrate(user_input, k=4)
    prompt = "\n".join(hydrated)
    return f"{prompt}\n\nUser: {user_input}\nAda:"

# Run interaction
user_input = input("Ask Ada something: ")
final_prompt = build_prompt(user_input)
response = llm.invoke(final_prompt)
print("\nAda:\n", response.content)
