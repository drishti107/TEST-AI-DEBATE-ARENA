import streamlit as st
import uuid
import time
import pandas as pd
import plotly.graph_objects as go
from gtts import gTTS
from io import BytesIO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Debate Arena 2.0", page_icon="⚔️", layout="wide")

# --- CSS STYLING ---
st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #00FF41; }
    .ai-health > div > div > div > div { background-color: #FF4B4B; }
    .hud-container { background-color: #1E1E1E; padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #333; }
    .stat-box { text-align: center; }
    .crowd-reaction { font-style: italic; color: #FFD700; text-align: center; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

# --- API KEY ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = "PASTE_YOUR_KEY_HERE"

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
                model="gemini-2.0-flash", 
                google_api_key=GOOGLE_API_KEY,
                temperature=0.7
            )
        except Exception as e:
            st.error(f"Initialization Error: {e}")

    def speak(self, text):
        """Converts text to audio bytes"""
        try:
            tts = gTTS(text=text, lang='en')
            fp = BytesIO()
            tts.write_to_fp(fp)
            return fp
        except:
            return None

    def generate_rebuttal(self, topic, argument, history, persona, stance):
        # Flatten history
        hist_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-4:]])
        template = """
        You are {persona}. Topic: {topic}. Stance: {stance}.
        History: {hist_text}
        Opponent says: "{argument}"
        
        Reply directly. Be sharp, witty, and logical. 
        Max 3 sentences.
        """
        try:
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm
            res = chain.invoke({"persona": persona, "topic": topic, "stance": stance, "hist_text": hist_text, "argument": argument})
            return res.content
        except: return "I disagree."

    def judge_turn(self, topic, user_arg, ai_arg):
        template = """
        Judge this debate turn. Topic: {topic}.
        A: "{user_arg}"
        B: "{ai_arg}"
        
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

# --- HELPER: PLOTLY CHART ---
def plot_debate_flow(score_history):
    if not score_history: return None
    
    df = pd.DataFrame(score_history)
    df['Turn'] = range(1, len(df) + 1)
    
    # Calculate Momentum (Cumulative Logic Difference)
    df['Momentum'] = df['user_score'] - df['ai_score']
    
    fig = go.Figure()
    
    # Add Zero Line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
    
    # Plot Momentum
    fig.add_trace(go.Scatter(
        x=df['Turn'], y=df['Momentum'],
        mode='lines+markers',
        name='Advantage',
        line=dict(color='#00FF41', width=3),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title="⚔️ Momentum of Debate (Logic Flow)",
        xaxis_title="Turn Number",
        yaxis_title="Advantage (User vs AI)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# --- SESSION STATE ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.score_history = [] # For the chart
    st.session_state.user_hp = 100
    st.session_state.ai_hp = 100
    st.session_state.started = False
    st.session_state.crowd_text = "The audience is waiting..."

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Arena 2.0")
    mode = st.radio("Mode:", ["User vs AI", "AI vs AI (Simulation)"])
    enable_audio = st.toggle("Enable Voice Output 🔊", value=True)
    
    st.divider()
    topic = st.text_input("Topic:", "Universal Basic Income")
    
    if mode == "User vs AI":
        persona = st.selectbox("Opponent:", ["Logical Vulcan", "Sarcastic Troll", "Philosopher"])
        ai_side = st.radio("AI Stance:", ["Against", "For"])
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
st.title("⚔️ AI Debate Arena 2.0")

if not st.session_state.started:
    st.info("👈 Configure the arena to begin.")
    st.stop()

# --- HUD (Heads Up Display) ---
if st.session_state.mode == "User":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.metric("Your Health", f"{st.session_state.user_hp}%", delta=None)
        st.progress(st.session_state.user_hp/100)
    with col2:
        st.markdown(f"<div class='crowd-reaction'>📢 {st.session_state.crowd_text}</div>", unsafe_allow_html=True)
        # Render Chart
        if st.session_state.score_history:
            fig = plot_debate_flow(st.session_state.score_history)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    with col3:
        st.metric(f"{st.session_state.persona} Health", f"{st.session_state.ai_hp}%")
        st.progress(st.session_state.ai_hp/100)
    st.divider()

# --- MODE 1: USER VS AI ---
if st.session_state.mode == "User":
    
    # Game Over
    if st.session_state.user_hp <= 0 or st.session_state.ai_hp <= 0:
        winner = "YOU" if st.session_state.user_hp > 0 else "AI"
        st.balloons() if winner == "YOU" else None
        st.error(f"GAME OVER. Winner: {winner}")
        
        if st.button("Analyze Match"):
            rep = engine.generate_report(st.session_state.messages, st.session_state.topic)
            st.markdown(f"**Coaching:** {rep.improvement_tips[0]}")
        st.stop()

    # Chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "audio" in msg and msg["audio"]:
                st.audio(msg["audio"], format="audio/mp3")

    # Input
    if prompt := st.chat_input("Make your point..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner(f"{st.session_state.persona} is thinking..."):
                # 1. Generate Rebuttal
                rebuttal = engine.generate_rebuttal(
                    st.session_state.topic, prompt, st.session_state.messages, 
                    st.session_state.persona, st.session_state.ai_side
                )
                
                # 2. Audio Generation
                audio_fp = None
                if enable_audio:
                    audio_fp = engine.speak(rebuttal)
                
                st.write(rebuttal)
                if audio_fp: st.audio(audio_fp, format='audio/mp3')
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": rebuttal,
                    "audio": audio_fp
                })
                
                # 3. Judge
                score = engine.judge_turn(st.session_state.topic, prompt, rebuttal)
                
                # Update Chart Data
                st.session_state.score_history.append({
                    "user_score": score.user_logic,
                    "ai_score": score.ai_logic
                })
                
                # Logic Calculation
                dmg = 0
                if score.winner == "ai":
                    dmg = (score.ai_logic - score.user_logic) / 2
                    st.session_state.user_hp = max(0, int(st.session_state.user_hp - dmg))
                    st.session_state.crowd_text = f"Ouch! The judge cites {score.fallacies_detected} in your argument!"
                elif score.winner == "user":
                    dmg = (score.user_logic - score.ai_logic) / 2
                    st.session_state.ai_hp = max(0, int(st.session_state.ai_hp - dmg))
                    st.session_state.crowd_text = "The crowd cheers! Superior logic detected!"
                else:
                    st.session_state.crowd_text = "A stalemate! Both sides holding ground."

                st.rerun()

# --- MODE 2: AI VS AI SIMULATION ---
elif st.session_state.mode == "Sim":
    st.subheader(f"🍿 Spectator Mode: {st.session_state.p1} vs {st.session_state.p2}")
    
    # Placeholder for the chart to update in real-time
    chart_spot = st.empty()
    chat_spot = st.container()
    
    if st.session_state.sim_active:
        history = []
        
        # Round 1: P1 Opens
        with chat_spot:
            with st.chat_message("user", avatar="🔵"):
                opening = engine.generate_rebuttal(st.session_state.topic, "Start debate", [], st.session_state.p1, "For")
                st.write(f"**{st.session_state.p1}:** {opening}")
                history.append({"role": "user", "content": opening})
        
        prev_arg = opening
        
        # Loop for 4 Rounds
        for i in range(4):
            time.sleep(1)
            
            # P2 Rebuts
            with chat_spot:
                with st.chat_message("assistant", avatar="🔴"):
                    with st.spinner(f"{st.session_state.p2} is retorting..."):
                        reb_2 = engine.generate_rebuttal(st.session_state.topic, prev_arg, history, st.session_state.p2, "Against")
                        st.write(f"**{st.session_state.p2}:** {reb_2}")
                        history.append({"role": "assistant", "content": reb_2})
            
            # Judge P2's Move for Chart
            score = engine.judge_turn(st.session_state.topic, prev_arg, reb_2)
            st.session_state.score_history.append({"user_score": score.user_logic, "ai_score": score.ai_logic})
            
            # Update Chart Live
            with chart_spot:
                fig = plot_debate_flow(st.session_state.score_history)
                st.plotly_chart(fig, use_container_width=True)
            
            prev_arg = reb_2
            time.sleep(1)
            
            # P1 Rebuts Back (unless last round)
            if i < 3:
                with chat_spot:
                    with st.chat_message("user", avatar="🔵"):
                        with st.spinner(f"{st.session_state.p1} is thinking..."):
                            reb_1 = engine.generate_rebuttal(st.session_state.topic, prev_arg, history, st.session_state.p1, "For")
                            st.write(f"**{st.session_state.p1}:** {reb_1}")
                            history.append({"role": "user", "content": reb_1})
                
                # Judge P1's Move
                score = engine.judge_turn(st.session_state.topic, reb_1, prev_arg)
                # Flip logic scores because P1 is 'user' in our schema
                st.session_state.score_history.append({"user_score": score.ai_logic, "ai_score": score.user_logic}) 
                
                with chart_spot:
                    fig = plot_debate_flow(st.session_state.score_history)
                    st.plotly_chart(fig, use_container_width=True)
                    
                prev_arg = reb_1

        st.session_state.sim_active = False
        st.success("Simulation Finished!")
        if st.button("Clear Arena"):
            st.session_state.started = False
            st.rerun()