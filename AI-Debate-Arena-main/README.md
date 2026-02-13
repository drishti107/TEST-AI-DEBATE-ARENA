#  Team Bitwise - Dev Heat IIIT Surat

<div align="center">
  <img src="https://img.shields.io/badge/Made%20with-Python-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/AI-Gemini%202.5-orange?style=for-the-badge&logo=google" alt="Gemini">
</div>

<br />

> **A Next-Gen Gamified Debate Simulator built for the Dev Heat IIIT Surat hackathon.**

---

##  About
This project leverages the power of **Google Gemini 2.5 Flash** combined with **LangChain** to create a competitive, real-time debate simulator.

The system is designed to gamify critical thinking by pitting users against adaptive AI personas. It processes arguments in real-time, evaluates them using a strictly impartial "AI Judge," and provides instant feedback via a "Health Point" (HP) system. By simulating high-stakes verbal combat, we demonstrate how GenAI can be used for educational reinforcement and logic training.

**Key Features:**
*  **Gamified Combat:** Users and AI have Health Bars; weak logic or irrelevance deals damage.
*  **Adaptive Personas:** Debate against distinct personalities like a "Logical Vulcan" or "Aggressive Troll."
*  **Real-time Judging:** Instant scoring of Logic, Relevance, and Civility for every turn.
*  **Post-Match Analysis:** A comprehensive coaching report generated automatically after the match.

---

##  Technology Stack
We used a modern tech stack focused on speed, interactivity, and AI reasoning:

* **Core Engine:** [LangChain](https://www.langchain.com/) (Orchestration & Prompt Chaining)
* **Language:** Python
* **Frontend:** Streamlit
* **AI/LLM:** Google Gemini 2.5 Flash (via API)
* **Validation:** Pydantic (Structured Output Parsing)

---

##  Repo Structure
```bash
AI-DEBATE-ARENA/
├── streamlit_app.py    # Main application logic (Frontend + AI Backend)
├── requirements.txt    # List of project dependencies
└── README.md           # Project documentation
```
## How to Clone
```bash
git clone https://github.com/arush-07/AI-DEBATE-ARENA.git
cd AI-DEBATE-ARENA
```

---

## Install Dependencies

Make sure you have **Python 3.9+** installed.

```bash
pip install -r requirements.txt
```

---

## Set Up Environment Variables

You need a **Google Gemini API Key**.

### Option 1: Set as Environment Variable (Recommended)

**Windows (PowerShell):**
```powershell
setx GOOGLE_API_KEY "your_api_key_here"
```

**Mac/Linux:**
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

### Option 2: Create a `.env` File

Create a `.env` file in the root directory:

```
GOOGLE_API_KEY=your_api_key_here
```

---

## Run the Application

```bash
streamlit run streamlit_app.py
```

Your app will open automatically in your browser 

##  Team Bitwise
Made by:

- **Arush Pradhan**
- **Drishti Verma**
- **Aarav Parikh**

<div align="center">
  <i>Built for the IIIT Surat hackathon.</i>
</div>
