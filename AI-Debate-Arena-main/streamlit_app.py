import streamlit as st
import uuid
import time
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import random
from PIL import Image
from gtts import gTTS
from io import BytesIO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Debate Arena 16.0", page_icon="⚔️", layout="wide")

# --- CSS STYLING ---
st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #00FF41; }
    .ai-health > div > div > div > div { background-color: #FF4B4B; }
    .league-bronze { color: #CD7F32; font-weight: bold; }
    .league-silver { color: #C0C0C0; font-weight: bold; }
    .league-gold { color: #FFD700; font-weight: bold; }
    .league-platinum { color: #E5E4E2; font-weight: bold; }
    .league-diamond { color: #B9F2FF; font-weight: bold; }
    .stat-card { background-color: #1E1E1E; padding: 15px; border-radius: 10px; border: 1px solid #333; text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- API KEYS SETUP ---
try: GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except: GOOGLE_API_KEY = None
try: HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except: HF_TOKEN = None
try: GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except: GROQ_API_KEY = None

# --- DATA MODELS ---
class TurnScore(BaseModel):
    user_logic: int = Field(..., description="0-100 score for logic")
    ai_logic: int = Field(..., description="0-100 score for logic")
    winner: str = Field(..., description="'user', 'ai', or 'draw'")
    reasoning: str = Field(..., description="Brief reason for the score")
    fallacies_detected: str = Field(..., description="Name any logical fallacies used (or 'None')")

class FinalAnalysis(BaseModel):
    winner: str
    best_point_user: str
    weakest_point_user: str
    improvement_tips: List[str]

# --- AI ENGINE ---
class DebateEngine:
    def __init__(self, provider, api_key):
        self.provider, self.api_key = provider, api_key
        try:
            if provider == "Google Gemini":
                genai.configure(api_key=api_key)
                self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
            elif provider == "Groq (Fastest)":
                self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
            elif provider == "Hugging Face":
                self.llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta", huggingfacehub_api_token=api_key, task="text-generation")
        except Exception as e: st.error(f"Error: {e}")

    def generate_response(self, topic, argument, history, persona, stance, image_data=None):
        if image_data and self.provider == "Google Gemini":
            model = genai.GenerativeModel("gemini-1.5-flash")
            res = model.generate_content([f"Refute this: {argument}", image_data])
            return res.text
        
        hist_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-3:]])
        template = f"You are {{persona}}. Stance: {{stance}}. Topic: {{topic}}. History: {{hist_text}}. Refute the argument: {{argument}}. Max 3 sentences."
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm
        res = chain.invoke({"persona": persona, "topic": topic, "stance": stance, "hist_text": hist_text, "argument": argument})
        return res.content if hasattr(res, 'content') else str(res)

    def judge_turn(self, topic, user_arg, ai_arg):
        if self.provider in ["Google Gemini", "Groq (Fastest)"]:
            template = "Judge logic (0-100). Topic: {topic}. User: {user_arg}. AI: {ai_arg}."
            structured = self.llm.with_structured_output(TurnScore)
            prompt = ChatPromptTemplate.from_template(template)
            return (prompt | structured).invoke({"topic": topic, "user_arg": user_arg, "ai_arg": ai_arg})
        return TurnScore(user_logic=random.randint(40, 80), ai_logic=60, winner="draw", reasoning="Manual", fallacies_detected="N/A")

    def generate_report(self, history, topic):
        if self.provider in ["Google Gemini", "Groq (Fastest)"]:
            template = "Analyze the debate: {topic}. Provide feedback for the user."
            structured = self.llm.with_structured_output(FinalAnalysis)
            prompt = ChatPromptTemplate.from_template(template)
            return (prompt | structured).invoke({"history": str(history), "topic": topic})
        return None

# --- GAMIFICATION LOGIC ---
def get_league(xp):
    if xp < 500: return ("Bronze", "league-bronze")
    if xp < 1500: return ("Silver", "league-silver")
    if xp < 3500: return ("Gold", "league-gold")
    if xp < 7000: return ("Platinum", "league-platinum")
    return ("Diamond", "league-diamond")

# --- SESSION STATE ---
if "xp" not in st.session_state: st.session_state.update({"xp": 0, "level": 1, "wins": 0, "streak": 0, "messages": [], "user_hp": 100, "ai_hp": 100, "started": False})

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Profile & League")
    league_name, league_css = get_league(st.session_state.xp)
    st.markdown(f"League: <span class='{league_css}'>{league_name}</span>", unsafe_allow_html=True)
    st.progress(min(1.0, (st.session_state.xp % 500) / 500))
    
    col1, col2 = st.columns(2)
    col1.metric("XP", st.session_state.xp)
    col2.metric("Wins", st.session_state.wins)
    
    st.divider()
    provider = st.selectbox("Brain", ["Groq (Fastest)", "Google Gemini", "Hugging Face"])
    api_key = GOOGLE_API_KEY if provider=="Google Gemini" else (GROQ_API_KEY if provider=="Groq (Fastest)" else HF_TOKEN)
    if not api_key: api_key = st.text_input(f"{provider} Key", type="password")
    
    if api_key: engine = DebateEngine(provider, api_key)
    else: st.stop()
    
    topic = st.text_input("Topic", "AI is a threat")
    if st.button("Battle! 🔥"):
        st.session_state.update({"messages": [], "user_hp": 100, "ai_hp": 100, "started": True, "topic": topic})
        st.rerun()

# --- MAIN UI ---
st.title("⚔️ AI Debate Arena")

if st.session_state.started:
    c1, c2, c3 = st.columns([1, 2, 1])
    c1.metric("User HP", f"{st.session_state.user_hp}%")
    c3.metric("AI HP", f"{st.session_state.ai_hp}%")
    
    # Game Loop
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])
    
    if prompt := st.chat_input("Your argument..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = engine.generate_response(st.session_state.topic, prompt, st.session_state.messages, "Vulcan", "Against")
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Scoring & XP
                score = engine.judge_turn(st.session_state.topic, prompt, response)
                st.session_state.xp += score.user_logic # Logic XP
                
                # Damage
                if score.winner == "user": st.session_state.ai_hp -= (score.user_logic - score.ai_logic)
                else: st.session_state.user_hp -= (score.ai_logic - score.user_logic)
                
                if st.session_state.ai_hp <= 0:
                    st.session_state.wins += 1
                    st.session_state.xp += 500 # Win Bonus
                    st.balloons()
                    st.session_state.started = False
                
                st.rerun()