import streamlit as st
import uuid
import time
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
from gtts import gTTS
from io import BytesIO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Debate Arena 3.1", page_icon="🎤", layout="wide")

# --- CSS STYLING ---
st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #00FF41; }
    .ai-health > div > div > div > div { background-color: #FF4B4B; }
    .crowd-reaction { font-style: italic; color: #FFD700; text-align: center; font-size: 0.9em; }
    .history-box { font-family: monospace; font-size: 0.8em; white-space: pre-wrap; background: #222; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- API KEY ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = "PASTE_YOUR_KEY_HERE"

# Configure Raw Gemini for Audio Transcription
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
            # Using 1.5-Flash for maximum speed and stability
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                google_api_key=GOOGLE_API_KEY,
                temperature=0.7
            )
        except Exception as e:
            st.error(f"Initialization Error: {e}")

    def speak(self, text):
        """Converts text to audio bytes"""
        try:
            if not text: return None
            tts = gTTS(text=text, lang='en')
            fp = BytesIO()
            tts.write_to_fp(fp)
            return fp
        except:
            return None

    def transcribe_audio(self, audio_file):
        """Uses Gemini to transcribe audio bytes to text"""
        try:
            # FIXED: Trying a more standard model name if the specific one fails
            model = genai.GenerativeModel("gemini-2.5-flash") 
            audio_bytes = audio_file.read()
            prompt = "Transcribe this audio exactly as spoken. Do not add any commentary."
            response = model.generate_content([
                prompt,
                {"mime_type": "audio/mp3", "data": audio_bytes}
            ])
            return response.text
        except Exception as e:
            # Fallback for transcription errors
            st.error(f"Transcription Error: {e}")
            return None

    def _safe_invoke(self, chain, inputs, retries=3):
        """Helper to handle Rate Limits (429 Errors)"""
        for i in range(retries):
            try:
                return chain.invoke(inputs)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    time.sleep(4 * (i + 1))
                else:
                    return None
        return None

    def generate_opening(self, topic, persona, stance):
        template = "You are {persona}. Topic: {topic}. Stance: {stance}. Generate a provocative 2-sentence opening argument."
        try:
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm
            res = self._safe_invoke(chain, {"persona": persona, "topic": topic, "stance": stance})
            return res.content if res else "Let's debate."
        except: return "I am ready."

    def generate_rebuttal(self, topic, argument, history, persona, stance):
        hist_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-4:]])
        template = """
        You are {persona}. Topic: {topic}. Stance: {stance}.
        History: {hist_text}
        Opponent says: "{argument}"
        Reply directly. Be sharp, witty, and logical. Max 3 sentences.
        """
        try:
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm
            res = self._safe_invoke(chain, {"persona": persona, "topic": topic, "stance": stance, "hist_text": hist_text, "argument": argument})
            return res.content if res else "I disagree."
        except: return "I disagree."

    def judge_turn(self, topic, user_arg, ai_arg):
        template = "Judge this debate turn. Topic: {topic}. A: '{user_arg}' B: '{ai_arg}'. Score logic (0-100). Identify fallacies."
        try:
            structured = self.llm.with_structured_output(TurnScore)
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | structured
            res = self._safe_invoke(chain, {"topic": topic, "user_arg": user_arg, "ai_arg": ai_arg})
            return res if res else TurnScore(user_logic=50, ai_logic=50, winner="draw", reasoning="Timeout", fallacies_detected="None")
        except: return TurnScore(user_logic=50, ai_logic=50, winner="draw", reasoning="Error", fallacies_detected="None")

    def generate_report(self, history, topic):
        hist_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        template = "Analyze debate: {topic}\n{history}\nProvide coaching report."
        try:
            structured = self.llm.with_structured_output(FinalAnalysis)
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | structured
            return self._safe_invoke(chain, {"history": hist_text, "topic": topic})
        except: return None

engine = DebateEngine()

# --- HELPER: PLOTLY CHART ---
def plot_debate_flow(score_history):
    if not score_history: return None
    df = pd.DataFrame(score_history)
    df['Turn'] = range(1, len(df) + 1)
    df['Momentum'] = df['user_score'] - df['ai_score']
    
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_trace(go.Scatter(x=df['Turn'], y=df['Momentum'], mode='lines+markers', name='Advantage', line=dict(color='#00FF41', width=3), fill='tozeroy'))
    fig.update_layout(title="⚔️ Debate Momentum", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# --- SESSION STATE ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.score_history = []
    st.session_state.user_hp = 100
    st.session_state.ai_hp = 100
    st.session_state.started = False
    st.session_state.crowd_text = "The audience is waiting..."

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Arena 3.1")
    mode = st.radio("Mode:", ["User vs AI", "AI vs AI (Simulation)"])
    enable_audio = st.toggle("Enable AI Voice 🔊", value=True)
    
    st.divider()
    
    # --- NEW FEATURE: HISTORY & LOGS ---
    with st.expander("📜 Debate History & Logs"):
        if st.session_state.messages:
            log_text = f"TOPIC: {st.session_state.get('topic', 'N/A')}\n\n"
            for msg in st.session_state.messages:
                role = "YOU" if msg['role'] == "user" else "AI"
                log_text += f"[{role}]: {msg['content']}\n\n"
            
            st.download_button("💾 Download Transcript", log_text, file_name="debate_log.txt")
            st.text_area("Live Log", log_text, height=200, disabled=True)
        else:
            st.caption("Start a debate to see logs.")

    st.divider()
    topic = st.text_input("Topic:", "Universal Basic Income")
    
    if mode == "User vs AI":
        persona = st.selectbox("Opponent:", ["Logical Vulcan", "Sarcastic Troll", "Philosopher"])
        ai_side = st.radio("AI Stance:", ["Against", "For"])
        
        # --- NEW FEATURE: AI MOVES FIRST ---
        who_starts = st.radio("Who starts?", ["Me (User)", "AI (Opponent)"], index=0)
        
        if st.button("Start Debate 🔥", use_container_width=True):
            st.session_state.messages = []
            st.session_state.score_history = []
            st.session_state.user_hp = 100
            st.session_state.ai_hp = 100
            st.session_state.started = True
            st.session_state.mode = "User"
            st.session_state.persona = persona
            st.session_state.topic = topic
            st.session_state.ai_side = ai_side
            
            # Logic for AI Starting First
            if who_starts == "AI (Opponent)":
                 with st.spinner(f"{persona} is preparing an opening statement..."):
                    opening = engine.generate_opening(topic, persona, ai_side)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": opening, 
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
            st.session_state.topic = topic
            st.session_state.sim_active = True
            st.rerun()

# --- MAIN UI ---
st.title("⚔️ AI Debate Arena 3.1")

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
        if st.session_state.score_history:
            fig = plot_debate_flow(st.session_state.score_history)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
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

    # Display Chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "audio" in msg and msg["audio"]:
                st.audio(msg["audio"], format="audio/mp3")

    # --- INPUT AREA (TEXT OR VOICE) ---
    st.markdown("### Make your move")
    
    # Text Input first (to ensure layout stability)
    text_input = st.chat_input("...or type your argument here")

    # Voice Input
    voice_input = st.audio_input("🎤 Tap to Speak")

    # Logic to handle inputs
    final_prompt = None
    
    if voice_input:
        with st.spinner("Transcribing your voice..."):
            final_prompt = engine.transcribe_audio(voice_input)
    elif text_input:
        final_prompt = text_input

    if final_prompt:
        st.session_state.messages.append({"role": "user", "content": final_prompt})
        
        # AI Turn
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
                
                # Scoring
                score = engine.judge_turn(st.session_state.topic, final_prompt, rebuttal)
                st.session_state.score_history.append({"user_score": score.user_logic, "ai_score": score.ai_logic})
                
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
                
                # Rerun to update the UI
                st.rerun()

# --- MODE 2: AI VS AI SIMULATION ---
elif st.session_state.mode == "Sim":
    st.subheader(f"🍿 Spectator Mode: {st.session_state.p1} vs {st.session_state.p2}")
    
    chart_spot = st.empty()
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
            
            score = engine.judge_turn(st.session_state.topic, prev_arg, reb_2)
            st.session_state.score_history.append({"user_score": score.user_logic, "ai_score": score.ai_logic})
            
            with chart_spot:
                fig = plot_debate_flow(st.session_state.score_history)
                st.plotly_chart(fig, use_container_width=True)
            
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
                
                score = engine.judge_turn(st.session_state.topic, reb_1, prev_arg)
                st.session_state.score_history.append({"user_score": score.ai_logic, "ai_score": score.user_logic}) 
                
                with chart_spot:
                    fig = plot_debate_flow(st.session_state.score_history)
                    st.plotly_chart(fig, use_container_width=True)
                    
                prev_arg = reb_1

        progress_bar.empty()
        st.session_state.sim_active = False
        st.balloons()
        st.success("Simulation Finished!")
        if st.button("Clear Arena"):
            st.session_state.started = False
            st.rerun()