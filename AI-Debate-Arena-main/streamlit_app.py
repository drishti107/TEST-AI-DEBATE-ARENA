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
st.set_page_config(page_title="AI Debate Arena 5.0", page_icon="⚔️", layout="wide")

# --- CSS STYLING ---
st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #00FF41; }
    .ai-health > div > div > div > div { background-color: #FF4B4B; }
    .crowd-reaction { font-style: italic; color: #FFD700; text-align: center; font-size: 0.9em; }
    .stat-box { background: #1E1E1E; padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 10px; }
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
    best_point_user: str
    weakest_point_user: str
    improvement_tips: List[str]

# --- AI ENGINE ---
class DebateEngine:
    def __init__(self):
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                google_api_key=GOOGLE_API_KEY,
                temperature=0.7
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
        Generate a professional but provocative 2-sentence opening argument.
        """
        try:
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm
            res = chain.invoke({"persona": persona, "topic": topic, "stance": stance})
            return res.content
        except: return "Let's begin."

    def generate_rebuttal(self, topic, argument, history, persona, stance):
        hist_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-4:]])
        
        template = f"""
        You are {{persona}}. Topic: {{topic}}. Stance: {{stance}}.
        History: {{hist_text}}
        Opponent says: "{{argument}}"
        
        Reply directly. Be sharp, witty, and logical. 
        Max 3 sentences.
        """
        try:
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm
            res = chain.invoke({"persona": persona, "topic": topic, "stance": stance, "hist_text": hist_text, "argument": argument})
            return res.content
        except: return "I disagree."

    def judge_turn(self, topic, user_arg, ai_arg, time_taken):
        penalty_text = ""
        if time_taken > 20:
            penalty_text = f"NOTE: The User took {time_taken} seconds to reply. PENALIZE their logic score slightly for hesitation."
            
        template = f"""
        Judge turn. Topic: {{topic}}.
        User ({time_taken}s delay): "{{user_arg}}"
        AI: "{{ai_arg}}"
        
        {penalty_text}
        
        Score logic (0-100). Identify fallacies.
        """
        try:
            structured = self.llm.with_structured_output(TurnScore)
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | structured
            return chain.invoke({"topic": topic, "user_arg": user_arg, "ai_arg": ai_arg})
        except: return TurnScore(user_logic=50, ai_logic=50, winner="draw", reasoning="Error", fallacies_detected="None")

    def generate_report(self, history, topic):
        hist_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        template = "Analyze debate: {topic}\n{history}\nProvide coaching report."
        try:
            structured = self.llm.with_structured_output(FinalAnalysis)
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | structured
            return chain.invoke({"history": hist_text, "topic": topic})
        except: return None

engine = DebateEngine()

# --- HELPER: RADAR CHART ---
def plot_radar(logic, speed_score, aggression):
    fig = go.Figure(data=go.Scatterpolar(
      r=[logic, speed_score, aggression],
      theta=['Logic', 'Speed', 'Confidence'],
      fill='toself',
      line_color='#00FF41'
    ))
    fig.update_layout(
      polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
      showlegend=False,
      paper_bgcolor='rgba(0,0,0,0)',
      plot_bgcolor='rgba(0,0,0,0)',
      font=dict(color='white'),
      height=200,
      margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig

# --- SESSION STATE ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.score_history = []
    st.session_state.user_hp = 100
    st.session_state.ai_hp = 100
    st.session_state.started = False
    st.session_state.crowd_text = "The arena is silent..."
    st.session_state.last_processed = "" 
    st.session_state.start_time = time.time()
    
    # Stats
    st.session_state.avg_logic = 50
    st.session_state.avg_speed = 50

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Arena 5.0")
    mode = st.radio("Mode:", ["User vs AI", "AI vs AI (Simulation)"])
    enable_audio = st.toggle("Enable AI Voice 🔊", value=True)
    
    st.divider()
    
    # ANALYTICS CHART
    if mode == "User vs AI" and st.session_state.started:
        st.caption("📊 Your Performance Radar")
        fig = plot_radar(st.session_state.avg_logic, st.session_state.avg_speed, 75)
        st.plotly_chart(fig, use_container_width=True)
    
    # HISTORY LOGS
    with st.expander("📜 Debate Logs"):
        if st.session_state.messages:
            log_text = f"TOPIC: {st.session_state.get('topic', 'N/A')}\n\n"
            for msg in st.session_state.messages:
                role = "YOU" if msg['role'] == "user" else "AI"
                log_text += f"[{role}]: {msg['content']}\n\n"
            st.download_button("💾 Download", log_text, file_name="debate_log.txt")
        else:
            st.caption("No history yet.")

    st.divider()
    
    # SURPRISE TOPIC
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        topic_input = st.text_input("Topic:", "Universal Basic Income")
    with col_t2:
        st.write("")
        st.write("")
        if st.button("🎲"):
            topics = ["Is cereal a soup?", "AI will replace teachers", "Cats are better than dogs", "Pineapple belongs on pizza", "Mars colonization is a waste"]
            topic_input = random.choice(topics)
            st.session_state.topic_preset = topic_input
            st.rerun()

    final_topic = st.session_state.get("topic_preset", topic_input)
    if "topic_preset" in st.session_state:
        st.info(f"Selected: {final_topic}")

    if mode == "User vs AI":
        # REMOVED "Roast Master" from this list
        persona = st.selectbox("Opponent:", ["Logical Vulcan", "Sarcastic Troll", "Philosopher", "Devil's Advocate"])
        ai_side = st.radio("AI Stance:", ["Against", "For"])
        who_starts = st.radio("Who starts?", ["Me (User)", "AI (Opponent)"], index=0)
        
        if st.button("Start Debate 🔥", use_container_width=True):
            st.session_state.messages = []
            st.session_state.score_history = []
            st.session_state.user_hp = 100
            st.session_state.ai_hp = 100
            st.session_state.started = True
            st.session_state.mode = "User"
            st.session_state.persona = persona
            st.session_state.topic = final_topic
            st.session_state.ai_side = ai_side
            st.session_state.last_processed = "" 
            st.session_state.start_time = time.time()
            st.session_state.avg_logic = 50
            
            if who_starts == "AI (Opponent)":
                 with st.spinner(f"{persona} is preparing..."):
                    opening = engine.generate_opening(final_topic, persona, ai_side)
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
            st.session_state.score_history = []
            st.session_state.started = True
            st.session_state.mode = "Sim"
            st.session_state.p1 = p1
            st.session_state.p2 = p2
            st.session_state.topic = final_topic
            st.session_state.sim_active = True
            st.rerun()

# --- MAIN UI ---
st.title("⚔️ AI Debate Arena 5.0")

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
    if st.session_state.user_hp <= 0 or st.session_state.ai_hp <= 0:
        winner = "YOU" if st.session_state.user_hp > 0 else "AI"
        st.balloons() if winner == "YOU" else None
        st.error(f"GAME OVER. Winner: {winner}")
        if st.button("Analyze Match"):
            rep = engine.generate_report(st.session_state.messages, st.session_state.topic)
            if rep: st.markdown(f"**Coaching:** {rep.improvement_tips[0]}")
        st.stop()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "audio" in msg and msg["audio"]:
                st.audio(msg["audio"], format="audio/mp3")

    # INPUT AREA
    st.markdown("### Make your move")
    
    # Calculate Time Since Last Turn
    time_elapsed = int(time.time() - st.session_state.start_time)
    
    text_input = st.chat_input("Type argument...")
    voice_input = st.audio_input("🎤 Tap to Speak")

    final_prompt = None
    if voice_input:
        with st.spinner("Transcribing..."):
            transcribed = engine.transcribe_audio(voice_input)
            if transcribed: final_prompt = transcribed
    elif text_input:
        final_prompt = text_input

    if final_prompt:
        if final_prompt == st.session_state.last_processed:
            pass
        else:
            st.session_state.last_processed = final_prompt
            st.session_state.messages.append({"role": "user", "content": final_prompt})
            
            # --- AI TURN ---
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
                    
                    # Score & Metrics
                    score = engine.judge_turn(st.session_state.topic, final_prompt, rebuttal, time_elapsed)
                    
                    # Update Stats
                    st.session_state.avg_logic = (st.session_state.avg_logic + score.user_logic) / 2
                    speed_val = max(0, 100 - (time_elapsed * 2))
                    st.session_state.avg_speed = (st.session_state.avg_speed + speed_val) / 2
                    
                    # Apply Damage
                    dmg = 0
                    if score.winner == "ai":
                        dmg = (score.ai_logic - score.user_logic) / 2
                        st.session_state.user_hp = max(0, int(st.session_state.user_hp - dmg))
                        st.session_state.crowd_text = f"Ouch! {score.fallacies_detected} detected!"
                    elif score.winner == "user":
                        dmg = (score.user_logic - score.ai_logic) / 2
                        st.session_state.ai_hp = max(0, int(st.session_state.ai_hp - dmg))
                        st.session_state.crowd_text = "Superior logic! Crowd cheers!"
                    else:
                        st.session_state.crowd_text = "Even exchange."
                    
                    # Reset Timer
                    st.session_state.start_time = time.time()
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
            time.sleep(2)
            
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
            time.sleep(2)
            
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