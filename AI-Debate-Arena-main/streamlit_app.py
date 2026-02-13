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
    .report-box { border: 1px solid #444; padding: 20px; border-radius: 10px; background-color: #222; }
</style>
""", unsafe_allow_html=True)

# --- API KEY SETUP ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = "PASTE_YOUR_KEY_HERE"

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
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", # Updated to latest fast model
                google_api_key=GOOGLE_API_KEY,
                temperature=0.8
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
            return "Let's debate."

    def generate_rebuttal(self, topic: str, argument: str, history: list, persona: str, difficulty: str, stance: str):
        # Flatten history for context
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-5:]])
        
        template = """
        Role: {role} ({difficulty} Mode)
        Topic: "{topic}"
        Stance: {stance}
        
        Conversation History:
        {history}
        
        Opponent's LATEST Argument: "{argument}"
        
        Task: 
        Directly rebut the opponent's latest argument. 
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
            return f"SYSTEM ERROR: {str(e)}"

    def judge_turn(self, topic: str, user_arg: str, ai_arg: str):
        template = """
        Act as an Impartial Debate Judge.
        Topic: {topic}
        
        User Argument: "{user_arg}"
        AI Rebuttal: "{ai_arg}"
        
        Task:
        1. Score User's Logic (0-100).
        2. Score AI's Logic (0-100).
        3. Score Relevance for both (0-100).
        4. Decide the winner based on logic scores.
        """
        try:
            structured_llm = self.llm.with_structured_output(TurnScore)
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | structured_llm
            return chain.invoke({"topic": topic, "user_arg": user_arg, "ai_arg": ai_arg})
        except Exception as e:
            return TurnScore(user_logic=50, user_relevance=50, ai_logic=50, ai_relevance=50, winner="draw", reasoning="Error")

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
    st.session_state.mode = "User vs AI" # Default mode

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Arena Setup")
    
    # NEW: Mode Selection
    mode = st.radio("Select Mode:", ["User vs AI (Game)", "AI vs AI (Watch & Learn)"])
    st.session_state.mode = mode
    
    st.divider()
    
    topic = st.text_input("Topic:", "AI will replace doctors")
    
    if mode == "User vs AI (Game)":
        col1, col2 = st.columns(2)
        with col1: persona = st.selectbox("Opponent:", ["Logical Vulcan", "Aggressive Troll", "Socratic Teacher"])
        with col2: difficulty = st.selectbox("Difficulty:", ["Easy", "Medium", "Hard"])
        ai_side = st.radio("AI's Stance:", ["AGAINST", "IN FAVOUR"], index=0)
        
        start_btn = st.button("Start Duel 🔥", use_container_width=True)
        
    else: # AI vs AI Mode
        st.info("Watch two AI personas debate to learn strategies.")
        col1, col2 = st.columns(2)
        with col1: 
            persona_1 = st.selectbox("Proponent (For):", ["Logical Vulcan", "Passionate Activist", "Elon Musk-esque"])
        with col2: 
            persona_2 = st.selectbox("Opponent (Against):", ["Socratic Teacher", "Aggressive Troll", "Shakespearean Poet"])
        
        difficulty = "Hard" # Force hard for better quality
        start_btn = st.button("Run Simulation 🎬", use_container_width=True)

    # Logic to Initialize Game/Sim
    if start_btn:
        st.session_state.messages = []
        st.session_state.user_hp = 100
        st.session_state.ai_hp = 100
        st.session_state.topic = topic
        st.session_state.started = True
        
        if mode == "User vs AI (Game)":
            st.session_state.persona = persona
            st.session_state.difficulty = difficulty
            st.session_state.ai_side = ai_side
            with st.spinner("AI Entering Arena..."):
                opening = engine.generate_opening(topic, persona, ai_side)
                st.session_state.messages.append({"role": "assistant", "content": opening})
            st.rerun()
            
        else: # AI vs AI Setup
            st.session_state.persona_1 = persona_1
            st.session_state.persona_2 = persona_2
            st.session_state.sim_running = True # Trigger the simulation loop

    # --- LIVE SCOREBOARD (Only for User Game) ---
    if st.session_state.started and mode == "User vs AI (Game)":
        st.divider()
        st.subheader("Live Health 🛡️ ")
        st.write(f"**You:** {st.session_state.user_hp}/100")
        st.progress(st.session_state.user_hp / 100)
        st.write(f"**Opponent:** {st.session_state.ai_hp}/100")
        st.progress(st.session_state.ai_hp / 100)
        
        st.divider()
        if st.button("QUIT ☠️", type="primary", use_container_width=True):
            st.session_state.user_hp = 0
            st.rerun()

# --- MAIN UI ---
st.title("⚔️ AI Debate Arena")

if not st.session_state.started:
    st.info("👈 Configure setup in the sidebar to begin.")
    st.stop()

# ==========================================
# MODE 1: USER VS AI (Original Game Logic)
# ==========================================
if st.session_state.mode == "User vs AI (Game)":

    # Game Over Logic
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
                with col1: st.success(f"**Best Point:**\n\n{report.best_point_user}")
                with col2: st.error(f"**Weakest Point:**\n\n{report.weakest_point_user}")
                st.subheader("💡 Coaching Tips")
                for tip in report.improvement_tips: st.info(f"• {tip}")
        st.stop()

    # Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input Loop
    if prompt := st.chat_input("Your argument..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                rebuttal = engine.generate_rebuttal(
                    st.session_state.topic, prompt, st.session_state.messages, 
                    st.session_state.persona, st.session_state.difficulty, st.session_state.ai_side
                )
                st.write(rebuttal)
                st.session_state.messages.append({"role": "assistant", "content": rebuttal})
                
                # Scoring
                score = engine.judge_turn(st.session_state.topic, prompt, rebuttal)
                user_dmg, ai_dmg = 0, 0
                
                if score.user_relevance < 50: user_dmg += 15
                if score.ai_relevance < 50: ai_dmg += 15
                logic_diff = score.user_logic - score.ai_logic
                
                if logic_diff > 0: ai_dmg += int(logic_diff / 2) 
                elif logic_diff < 0: user_dmg += int(abs(logic_diff) / 2)
                
                if score.user_logic > 85: 
                    ai_dmg += 10
                    st.toast("🔥 CRITICAL HIT! Superior logic!", icon="🔥")
                
                st.session_state.user_hp = max(0, st.session_state.user_hp - user_dmg)
                st.session_state.ai_hp = max(0, st.session_state.ai_hp - ai_dmg)
                
                if user_dmg > 0: st.toast(f"You lost {user_dmg} HP. {score.reasoning}", icon="💔")
                if ai_dmg > 0: st.toast(f"Opponent lost {ai_dmg} HP!", icon="🎯")
                
                time.sleep(1)
                st.rerun()

# ==========================================
# MODE 2: AI VS AI (Simulation Mode)
# ==========================================
elif st.session_state.mode == "AI vs AI (Watch & Learn)":
    
    st.subheader(f"🏟️ Exhibition Match: {st.session_state.persona_1} vs {st.session_state.persona_2}")
    st.caption(f"Topic: {st.session_state.topic}")
    
    # Container for the chat
    chat_container = st.container()
    
    # We use a button loop here to trigger the generation once
    if "sim_running" in st.session_state and st.session_state.sim_running:
        
        history = []
        
        # 1. Opening Statement (Proponent)
        with chat_container:
            with st.chat_message("user", avatar="🔵"): # Blue for Proponent
                with st.spinner(f"{st.session_state.persona_1} is opening..."):
                    opening = engine.generate_opening(st.session_state.topic, st.session_state.persona_1, "IN FAVOUR")
                    st.write(f"**{st.session_state.persona_1}:** {opening}")
                    history.append({"role": st.session_state.persona_1, "content": opening})
            
            last_arg = opening
            
            # 2. Debate Rounds (4 Rounds)
            rounds = 4
            for i in range(rounds):
                # Opponent Rebuts
                time.sleep(1.5)
                with st.chat_message("assistant", avatar="🔴"): # Red for Opponent
                    with st.spinner(f"{st.session_state.persona_2} is thinking..."):
                        rebuttal_con = engine.generate_rebuttal(
                            st.session_state.topic, last_arg, history, 
                            st.session_state.persona_2, "Hard", "AGAINST"
                        )
                        st.write(f"**{st.session_state.persona_2}:** {rebuttal_con}")
                        history.append({"role": st.session_state.persona_2, "content": rebuttal_con})
                        last_arg = rebuttal_con

                # Proponent Rebuts back
                if i < rounds - 1: # Don't rebut on the very last turn to end cleanly
                    time.sleep(1.5)
                    with st.chat_message("user", avatar="🔵"):
                        with st.spinner(f"{st.session_state.persona_1} is thinking..."):
                            rebuttal_pro = engine.generate_rebuttal(
                                st.session_state.topic, last_arg, history, 
                                st.session_state.persona_1, "Hard", "IN FAVOUR"
                            )
                            st.write(f"**{st.session_state.persona_1}:** {rebuttal_pro}")
                            history.append({"role": st.session_state.persona_1, "content": rebuttal_pro})
                            last_arg = rebuttal_pro

        st.session_state.sim_running = False # Stop loop
        st.success("✅ Simulation Complete!")
        
        # Generate Analysis of the Match
        with st.expander("📊 View Judge's Analysis", expanded=True):
            with st.spinner("Analyzing the debate logic..."):
                # We reuse the final report engine but tweak the prompt internally via context
                formatted_hist = [{"role": m["role"], "content": m["content"]} for m in history]
                report = engine.generate_final_report(formatted_hist, st.session_state.topic)
                
                if report:
                    st.markdown(f"""
                    ### 🏆 Winner: {report.winner}
                    
                    **Strongest Argument (Proponent):**
                    > {report.best_point_user}
                    
                    **Weakest Moment:**
                    > {report.weakest_point_user}
                    
                    **Key Takeaways for Students:**
                    """)
                    for tip in report.improvement_tips:
                        st.markdown(f"- {tip}")
    
    # If sim is done, offer reset
    if not st.session_state.sim_running:
         if st.button("Reset / New Simulation"):
             st.session_state.started = False
             st.rerun()