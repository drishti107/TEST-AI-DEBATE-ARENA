import streamlit as st
import uuid
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Debate Arena", page_icon="⚔️", layout="wide")

# --- CSS STYLING ---
st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #00FF41; }
    .ai-health > div > div > div > div { background-color: #FF4B4B; }
    .toast-popup { background-color: #333; color: white; }
</style>  
""", unsafe_allow_html=True)

# --- API KEY SETUP ---
try:
    # Try getting key from secrets first
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    # ⚠️ IF RUNNING LOCALLY, PASTE YOUR KEY BELOW ⚠️
    GOOGLE_API_KEY = "AIzaSyCzT3NMkwrYeySyH0cbnUtpBkSucgc_G94"

# --- DATA MODELS ---
class TurnScore(BaseModel):
    user_logic: int = Field(..., description="0-100 score for user logic")
    user_relevance: int = Field(..., description="0-100 score for user relevance")
    ai_logic: int = Field(..., description="0-100 score for AI rebuttal strength")
    ai_relevance: int = Field(..., description="0-100 score for AI directness")
    winner: str = Field(..., description="'user' or 'ai' or 'draw'")
    reasoning: str = Field(..., description="Why this side won the turn")

class FinalAnalysis(BaseModel):
    winner: str = Field(..., description="Overall winner")
    best_point_user: str = Field(..., description="The user's strongest argument")
    weakest_point_user: str = Field(..., description="The user's weakest moment")
    improvement_tips: List[str] = Field(..., description="3 specific tips to improve")

# --- AI ENGINE ---
class DebateEngine:
    def __init__(self):
        try:
            # FIXED: Changed model name to a valid one
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                google_api_key=GOOGLE_API_KEY,
                temperature=0.7
            )
        except Exception as e:
            st.error(f"Initialization Error: {e}")

    def generate_opening(self, topic: str, persona: str, stance: str):
        template = """
        You are debating as: {persona}. 
        Topic: "{topic}".
        Your Stance: You must ARGUE {stance} this topic.
        Generate a strong, 2-sentence opening argument.
        """
        try:
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm
            response = chain.invoke({"topic": topic, "persona": persona, "stance": stance})
            return response.content
        except Exception as e:
            st.error(f"Opening Gen Error: {e}")
            return "Let's debate."

    def generate_rebuttal(self, topic: str, argument: str, history: list, persona: str, difficulty: str, stance: str):
        # Format history specifically for context
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-5:]])
        
        template = """
        Role: {role} ({difficulty} Mode)
        Topic: "{topic}"
        Stance: {stance}
        
        Conversation History:
        {history}
        
        User's LATEST Argument: "{argument}"
        
        Task: 
        Directly rebut the user's latest argument. 
        Do not just say "I disagree". 
        Use logic and facts based on your persona.
        Keep it sharp (under 4 sentences).
        """
        try:
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm
            response = chain.invoke({
                "role": persona, "difficulty": difficulty, "topic": topic, 
                "history": history_text, "argument": argument, "stance": stance
            })
            return response.content
        except Exception as e:
            # FIXED: Show actual error
            return f"SYSTEM ERROR: {str(e)}"

    def judge_turn(self, topic: str, user_arg: str, ai_arg: str):
        template = """
        Act as an Impartial Debate Judge.
        Topic: {topic}
        
        User Argument: "{user_arg}"
        AI Rebuttal: "{ai_arg}"
        
        Task:
        1. Score User's Logic (0-100). Be generous if they make a good point.
        2. Score AI's Logic (0-100). Be STRICT.
        3. Score Relevance for both (0-100).
        4. Decide the winner based on logic scores.
        """
        try:
            structured_llm = self.llm.with_structured_output(TurnScore)
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | structured_llm
            return chain.invoke({"topic": topic, "user_arg": user_arg, "ai_arg": ai_arg})
        except Exception as e:
            st.error(f"Judging Error: {e}")
            return TurnScore(
                user_logic=50, user_relevance=50, 
                ai_logic=50, ai_relevance=50, 
                winner="draw", reasoning="Judge malfunction."
            )

    def generate_final_report(self, history: list, topic: str):
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        template = """
        Analyze this debate. Topic: {topic}. 
        History: {history}.
        Provide a coaching report for the User (Best point, Weakest point, 3 Tips).
        """
        try:
            structured_llm = self.llm.with_structured_output(FinalAnalysis)
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | structured_llm
            return chain.invoke({"history": history_text, "topic": topic})
        except Exception as e:
            st.error(f"Report Error: {e}")
            return None

# Initialize Engine
engine = DebateEngine()

# --- SESSION STATE ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.user_hp = 100
    st.session_state.ai_hp = 100
    st.session_state.started = False

if "ai_side" not in st.session_state:
    st.session_state.ai_side = "AGAINST the Topic"

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Arena Setup")
    topic = st.text_input("Topic:", "AI will replace doctors")
    col1, col2 = st.columns(2)
    with col1: persona = st.selectbox("Opponent:", ["Logical Vulcan", "Aggressive Troll", "Socratic Teacher"])
    with col2: difficulty = st.selectbox("Difficulty:", ["Easy", "Medium", "Hard"])
    ai_side = st.radio("AI's Stance:", ["AGAINST", "IN FAVOUR"], index=0)
    
    if st.button("Start Debate 🔥 ", use_container_width=True):
        st.session_state.messages = []
        st.session_state.user_hp = 100
        st.session_state.ai_hp = 100
        st.session_state.started = True
        st.session_state.topic = topic
        st.session_state.persona = persona
        st.session_state.difficulty = difficulty
        st.session_state.ai_side = ai_side
        
        with st.spinner("AI Entering Arena..."):
            opening = engine.generate_opening(topic, persona, ai_side)
            st.session_state.messages.append({"role": "assistant", "content": opening})
        st.rerun()

    # --- LIVE SCOREBOARD ---
    if st.session_state.started:
        st.divider()
        st.subheader("Live Health 🛡️ ")
        
        # User HP
        st.write(f"**You:** {st.session_state.user_hp}/100")
        st.progress(st.session_state.user_hp / 100)
        
        # AI HP
        st.write(f"**Opponent ({st.session_state.persona}):** {st.session_state.ai_hp}/100")
        st.progress(st.session_state.ai_hp / 100)
        
        st.divider()
        if st.button("QUIT ☠️", type="primary", use_container_width=True):
            st.session_state.user_hp = 0 # Force end
            st.rerun()

# --- MAIN UI ---
st.title("⚔️ AI Debate Arena")

if not st.session_state.started:
    st.info("👈 Configure setup in the sidebar to begin.")
    st.stop()

# --- GAME OVER LOGIC ---
if st.session_state.user_hp <= 0 or st.session_state.ai_hp <= 0:
    winner = "YOU" if st.session_state.user_hp > 0 else "AI"
    st.header(f"🏁 DEBATE OVER! Winner: {winner}")
    
    if st.button("Restart Debate"):
        st.session_state.started = False
        st.rerun()
        
    with st.spinner("Generating Analysis..."):
        report = engine.generate_final_report(st.session_state.messages, st.session_state.topic)
        if report:
            col1, col2 = st.columns(2)
            with col1: 
                st.success(f"**Best Point:**\n\n{report.best_point_user}")
            with col2: 
                st.error(f"**Weakest Point:**\n\n{report.weakest_point_user}")
            
            st.subheader("💡 Coaching Tips")
            for tip in report.improvement_tips: 
                st.info(f"• {tip}")
    st.stop()

# --- CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- GAME INPUT LOOP ---
if prompt := st.chat_input("Your argument..."):
    # 1. Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.write(prompt)

    # 2. Generate AI Rebuttal
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            rebuttal = engine.generate_rebuttal(
                st.session_state.topic, prompt, st.session_state.messages, 
                st.session_state.persona, st.session_state.difficulty, st.session_state.ai_side
            )
            st.write(rebuttal)
            st.session_state.messages.append({"role": "assistant", "content": rebuttal})
            
            # 3. Judge & Score
            score = engine.judge_turn(st.session_state.topic, prompt, rebuttal)
            
            user_dmg = 0
            ai_dmg = 0
            
            # Scoring Logic
            if score.user_relevance < 50: user_dmg += 15
            if score.ai_relevance < 50: ai_dmg += 15
            
            logic_diff = score.user_logic - score.ai_logic
            
            if logic_diff > 0:
                ai_dmg += int(logic_diff / 2) 
            elif logic_diff < 0:
                user_dmg += int(abs(logic_diff) / 2)
            
            # Critical Hit
            if score.user_logic > 85: 
                ai_dmg += 10
                st.toast("🔥 CRITICAL HIT! Superior logic!", icon="🔥")
            
            # Apply Damage
            st.session_state.user_hp = max(0, st.session_state.user_hp - user_dmg)
            st.session_state.ai_hp = max(0, st.session_state.ai_hp - ai_dmg)
            
            # Feedback Toasts
            if user_dmg > 0: 
                st.toast(f"You lost {user_dmg} HP. {score.reasoning}", icon="💔")
            if ai_dmg > 0: 
                st.toast(f"Opponent lost {ai_dmg} HP!", icon="🎯")
            
            time.sleep(1) 
            st.rerun()