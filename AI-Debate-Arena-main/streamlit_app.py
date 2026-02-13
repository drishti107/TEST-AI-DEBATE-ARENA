import streamlit as st
import uuid
import time
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import random
from gtts import gTTS
from io import BytesIO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Debate Arena 10.0", page_icon="⚔️", layout="wide")

# --- CSS STYLING ---
st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #00FF41; }
    .ai-health > div > div > div > div { background-color: #FF4B4B; }
    .crowd-reaction { font-style: italic; color: #FFD700; text-align: center; font-size: 1.1em; margin-top: 10px; }
    
    .report-card { background-color: #1E1E1E; padding: 20px; border-radius: 10px; border: 1px solid #333; margin-top: 20px; }
    .best-point { border-left: 4px solid #00FF41; padding-left: 10px; margin-bottom: 15px; }
    .worst-point { border-left: 4px solid #FF4B4B; padding-left: 10px; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# --- API KEY ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = "PASTE_YOUR_KEY_HERE"

genai.configure(api_key=GOOGLE_API_KEY)

# --- DATA MODELS ---
class TurnScore(BaseModel):
    user_logic: int = Field(..., description="0-100 score for logic")
    ai_logic: int = Field(..., description="0-100 score for logic")
    winner: str = Field(..., description="'user', 'ai', or 'draw'")
    reasoning: str = Field(..., description="Brief reason for the score")
    fallacies_detected: str = Field(..., description="Name any logical fallacies used (or 'None')")

class FinalAnalysis(BaseModel):
    winner: str
    best_point_user: str = Field(..., description="Quote the user's strongest argument")
    weakest_point_user: str = Field(..., description="Quote the user's weakest argument")
    improvement_tips: List[str] = Field(..., description="3 specific things the user should remember to improve")

# --- AI ENGINE ---
class DebateEngine:
    def __init__(self):
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                google_api_key=GOOGLE_API_KEY,
                temperature=0.8 # Slightly higher for creativity
            )
        except Exception as e:
            st.error(f"Initialization Error: {e}")

    def speak(self, text):
        try:
            if not text: return None
            tts = gTTS(text=text, lang='en')
            fp = BytesIO()
            tts.write_to_fp(fp)
            return fp
        except: return None

    def transcribe_audio(self, audio_file):
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            audio_bytes = audio_file.read()
            prompt = "Transcribe this audio exactly as spoken."
            response = model.generate_content([prompt, {"mime_type": "audio/mp3", "data": audio_bytes}])
            return response.text
        except: return None

    def generate_opening(self, topic, persona, stance):
        template = f"""
        You are {{persona}}. Topic: {{topic}}. Stance: {{stance}}.
        Generate a sharp, provocative 2-sentence opening argument.
        DO NOT be neutral. Pick a side and fight for it.
        """
        try:
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm
            res = chain.invoke({"persona": persona, "topic": topic, "stance": stance})
            return res.content
        except: return "I am ready to debate this topic."

    def generate_rebuttal(self, topic, argument, history, persona, stance):
        hist_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-4:]])
        template = f"""
        You are {{persona}}. Topic: {{topic}}. Stance: {{stance}}.
        History: {{hist_text}}
        Opponent says: "{{argument}}"
        
        TASK: Dissect the opponent's argument and provide a logical counter-point.
        CONSTRAINT: Never say "I disagree". Instead, explain WHY they are wrong.
        Keep it under 3 sentences.
        """
        try:
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm
            res = chain.invoke({"persona": persona, "topic": topic, "stance": stance, "hist_text": hist_text, "argument": argument})
            # Fallback if empty
            if not res.content: return "That logic is flawed. Consider the economic implications."
            return res.content
        except Exception as e: 
            # Better fallback than "I disagree"
            return f"That's an interesting perspective, but have you considered the long-term data regarding {topic}?"

    def judge_turn(self, topic, user_arg, ai_arg):
        template = f"""
        Judge turn. Topic: {{topic}}.
        User: "{{user_arg}}"
        AI: "{{ai_arg}}"
        
        Score logic (0-100) strictly based on facts and reasoning.
        """
        try:
            structured = self.llm.with_structured_output(TurnScore)
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | structured
            return chain.invoke({"topic": topic, "user_arg": user_arg, "ai_arg": ai_arg})
        except: return TurnScore(user_logic=50, ai_logic=50, winner="draw", reasoning="Error", fallacies_detected="None")

    def generate_report(self, history, topic):
        hist_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        template = """
        Analyze the full debate history. Topic: {topic}.
        History: {history}
        
        Task:
        1. Identify the Winner.
        2. Find the User's BEST point (quote it).
        3. Find the User's WEAKEST point (quote it).
        4. Provide 3 specific tips for the user to improve next time.
        """
        try:
            structured = self.llm.with_structured_output(FinalAnalysis)
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | structured
            return chain.invoke({"history": hist_text, "topic": topic})
        except: return None

engine = DebateEngine()

# --- CALLBACKS ---
def update_topic():
    topics = [
        "Is cereal a soup?", 
        "AI will replace teachers", 
        "Cats are better than dogs", 
        "Pineapple belongs on pizza", 
        "Mars colonization is a waste",
        "Social media does more harm than good",
        "Video games cause violence",
        "Messi or Ronaldo: Who's the GOAT?",
    ]
    st.session_state.topic_input = random.choice(topics)

# --- SESSION STATE ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.user_hp = 100
    st.session_state.ai_hp = 100
    st.session_state.started = False
    st.session_state.crowd_text = "The arena is silent..."
    st.session_state.last_processed = "" 
    st.session_state.audio_key = "audio_1"
    st.session_state.topic_input = "Universal Basic Income" # Default topic

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Arena 10.0")
    mode = st.radio("Mode:", ["User vs AI", "AI vs AI (Simulation)"])
    enable_audio = st.toggle("Enable AI Voice 🔊", value=True)
    
    st.divider()
    
    # HISTORY
    with st.expander("📜 Debate Logs"):
        if st.session_state.messages:
            log_text = f"TOPIC: {st.session_state.topic_input}\n\n"
            for msg in st.session_state.messages:
                role = "YOU" if msg['role'] == "user" else "AI"
                log_text += f"[{role}]: {msg['content']}\n\n"
            st.download_button("💾 Download", log_text, file_name="debate_log.txt")
        else:
            st.caption("No history yet.")

    st.divider()
    
    # TOPIC SECTION WITH AUTO-FILL
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        # Key is crucial here for the callback to work
        st.text_input("Topic:", key="topic_input")
    with col_t2:
        st.write("")
        st.write("")
        # Button updates session state via callback
        st.button("🎲", on_click=update_topic, help="Generate Random Topic")

    # QUIT BUTTON
    if mode == "User vs AI" and st.session_state.started:
        st.divider()
        if st.button("🏳️ Quit & Analyze", type="primary", use_container_width=True):
            st.session_state.user_hp = 0  # Instant Loss
            st.rerun()

    if mode == "User vs AI":
        persona = st.selectbox("Opponent:", ["Logical Vulcan", "Sarcastic Troll", "Philosopher", "Devil's Advocate"])
        ai_side = st.radio("AI Stance:", ["Against", "For"])
        who_starts = st.radio("Who starts?", ["Me (User)", "AI (Opponent)"], index=0)
        
        if st.button("Start Debate 🔥", use_container_width=True):
            st.session_state.messages = []
            st.session_state.user_hp = 100
            st.session_state.ai_hp = 100
            st.session_state.started = True
            st.session_state.mode = "User"
            st.session_state.persona = persona
            st.session_state.topic = st.session_state.topic_input # Use current input
            st.session_state.ai_side = ai_side
            st.session_state.last_processed = "" 
            st.session_state.audio_key = str(uuid.uuid4())
            
            if who_starts == "AI (Opponent)":
                 with st.spinner(f"{persona} is preparing..."):
                    opening = engine.generate_opening(st.session_state.topic_input, persona, ai_side)
                    st.session_state.messages.append({
                        "role": "assistant", "content": opening, 
                        "audio": engine.speak(opening) if enable_audio else None
                    })
            st.rerun()
            
    else: # AI vs AI
        p1 = st.selectbox("Proponent:", ["Elon Musk-esque", "Idealist Student"])
        p2 = st.selectbox("Opponent:", ["Grumpy Boomer", "Data Scientist"])
        if st.button("Run Simulation 🎬", use_container_width=True):
            st.session_state.messages = []
            st.session_state.started = True
            st.session_state.mode = "Sim"
            st.session_state.p1 = p1
            st.session_state.p2 = p2
            st.session_state.topic = st.session_state.topic_input
            st.session_state.sim_active = True
            st.rerun()

# --- MAIN UI ---
st.title("⚔️ AI Debate Arena 10.0")

if not st.session_state.started:
    st.info("👈 Configure the arena sidebar to begin.")
    st.stop()

# --- HUD ---
if st.session_state.mode == "User":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.metric("Your Health", f"{st.session_state.user_hp}%")
        st.progress(st.session_state.user_hp/100)
    with col2:
        st.markdown(f"<div class='crowd-reaction'>📢 {st.session_state.crowd_text}</div>", unsafe_allow_html=True)
    with col3:
        st.metric(f"{st.session_state.persona} Health", f"{st.session_state.ai_hp}%")
        st.progress(st.session_state.ai_hp/100)
    st.divider()

# --- MODE 1: USER VS AI ---
if st.session_state.mode == "User":
    
    # GAME OVER
    if st.session_state.user_hp <= 0 or st.session_state.ai_hp <= 0:
        winner = "YOU" if st.session_state.user_hp > 0 else "AI"
        
        if winner == "YOU":
            st.balloons()
            st.success(f"🏆 VICTORY! You defeated {st.session_state.persona}!")
        else:
            st.error(f"💀 DEFEAT! {st.session_state.persona} wins!")

        st.markdown("## 📊 Debate Analysis")
        with st.spinner("The judges are compiling your performance report..."):
            rep = engine.generate_report(st.session_state.messages, st.session_state.topic)
            
            if rep:
                st.markdown(f"""
                <div class="report-card">
                    <div class="best-point">
                        <strong>💎 Your Best Point:</strong><br>
                        <em>"{rep.best_point_user}"</em>
                    </div>
                    <div class="worst-point">
                        <strong>📉 Your Weakest Link:</strong><br>
                        <em>"{rep.weakest_point_user}"</em>
                    </div>
                    <strong>💡 Points to Remember (Coaching Tips):</strong>
                </div>
                """, unsafe_allow_html=True)
                
                for i, tip in enumerate(rep.improvement_tips):
                    st.info(f"{i+1}. {tip}")
            
        if st.button("Start New Debate"):
            st.session_state.started = False
            st.rerun()
            
        st.stop()

    # Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "audio" in msg and msg["audio"]:
                st.audio(msg["audio"], format="audio/mp3")

    # INPUT
    st.markdown("### Make your move")
    
    text_input = st.chat_input("Type argument...")
    voice_input = st.audio_input("🎤 Tap to Speak", key=st.session_state.audio_key)

    final_prompt = None
    if text_input:
        final_prompt = text_input
    elif voice_input:
        with st.spinner("Transcribing..."):
            transcribed = engine.transcribe_audio(voice_input)
            if transcribed and transcribed != st.session_state.last_processed:
                final_prompt = transcribed

    # Processing
    if final_prompt:
        if st.session_state.messages and st.session_state.messages[-1]['role'] == 'user':
            pass 
        else:
            st.session_state.last_processed = final_prompt
            st.session_state.messages.append({"role": "user", "content": final_prompt})
            
            # AI TURN
            with st.chat_message("assistant"):
                with st.spinner(f"{st.session_state.persona} is thinking..."):
                    
                    rebuttal = engine.generate_rebuttal(
                        st.session_state.topic, final_prompt, st.session_state.messages, 
                        st.session_state.persona, st.session_state.ai_side
                    )
                    
                    audio_fp = engine.speak(rebuttal) if enable_audio else None
                    st.write(rebuttal)
                    if audio_fp: st.audio(audio_fp, format='audio/mp3')
                    
                    st.session_state.messages.append({"role": "assistant", "content": rebuttal, "audio": audio_fp})
                    
                    # Score
                    score = engine.judge_turn(st.session_state.topic, final_prompt, rebuttal)
                    
                    # Damage
                    user_dmg = 0
                    ai_dmg = 0
                    
                    if score.winner == "ai":
                        user_dmg = int(score.ai_logic - score.user_logic)
                        if user_dmg < 10: user_dmg = 10
                        st.session_state.user_hp = max(0, st.session_state.user_hp - user_dmg)
                        st.session_state.crowd_text = f"Ouch! {score.fallacies_detected} detected!"
                        st.toast(f"💥 HIT! You lost {user_dmg} HP!", icon="🩸")
                        
                    elif score.winner == "user":
                        ai_dmg = int(score.user_logic - score.ai_logic)
                        if ai_dmg < 10: ai_dmg = 10
                        st.session_state.ai_hp = max(0, st.session_state.ai_hp - ai_dmg)
                        st.session_state.crowd_text = "Superior logic! Crowd cheers!"
                        st.toast(f"🎯 CRITICAL! AI lost {ai_dmg} HP!", icon="🔥")
                        
                    else:
                        st.session_state.crowd_text = "Even exchange."
                        st.toast("🛡️ Blocked! No damage.", icon="🛡️")
                    
                    time.sleep(0.5) # Reduced delay for speed
                    st.rerun()

# --- MODE 2: AI VS AI ---
elif st.session_state.mode == "Sim":
    st.subheader(f"🍿 Spectator Mode: {st.session_state.p1} vs {st.session_state.p2}")
    chat_spot = st.container()
    
    if st.session_state.sim_active:
        history = []
        with chat_spot:
            with st.chat_message("user", avatar="🔵"):
                placeholder = st.empty()
                placeholder.info(f"⏳ {st.session_state.p1} is opening...")
                opening = engine.generate_rebuttal(st.session_state.topic, "Start debate", [], st.session_state.p1, "For")
                placeholder.empty()
                st.write(f"**{st.session_state.p1}:** {opening}")
                history.append({"role": "user", "content": opening})
                st.session_state.messages.append({"role": "user", "content": f"{st.session_state.p1}: {opening}"})
        
        prev_arg = opening
        progress_bar = st.progress(0, text="Debate in progress...")
        
        for i in range(4):
            progress_bar.progress((i + 1) / 4)
            time.sleep(0.5) # FASTER SPEED
            
            with chat_spot:
                with st.chat_message("assistant", avatar="🔴"):
                    placeholder = st.empty()
                    placeholder.info(f"⏳ {st.session_state.p2} is reading...")
                    reb_2 = engine.generate_rebuttal(st.session_state.topic, prev_arg, history, st.session_state.p2, "Against")
                    placeholder.empty()
                    st.write(f"**{st.session_state.p2}:** {reb_2}")
                    history.append({"role": "assistant", "content": reb_2})
                    st.session_state.messages.append({"role": "assistant", "content": f"{st.session_state.p2}: {reb_2}"})
            
            prev_arg = reb_2
            time.sleep(0.5) # FASTER SPEED
            
            if i < 3:
                with chat_spot:
                    with st.chat_message("user", avatar="🔵"):
                        placeholder = st.empty()
                        placeholder.info(f"⏳ {st.session_state.p1} is thinking...")
                        reb_1 = engine.generate_rebuttal(st.session_state.topic, prev_arg, history, st.session_state.p1, "For")
                        placeholder.empty()
                        st.write(f"**{st.session_state.p1}:** {reb_1}")
                        history.append({"role": "user", "content": reb_1})
                        st.session_state.messages.append({"role": "user", "content": f"{st.session_state.p1}: {reb_1}"})
                prev_arg = reb_1

        progress_bar.empty()
        st.session_state.sim_active = False
        st.balloons()
        st.success("Simulation Finished!")
        if st.button("Clear Arena"):
            st.session_state.started = False
            st.rerun()