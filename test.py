import google.generativeai as genai 
import os 
genai.configure(api_key="2a2lVUoaHek89gjRdRrWMLXEGmxBLo8") 
try: 
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content("test") 
    print("PAID Gemini key is working!") 
except Exception as e: print(str(e))