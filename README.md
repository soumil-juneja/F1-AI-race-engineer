# 🏎️ F1 AI Race Engineer

An autonomous, agentic system designed for real-time motorsport analytics, strategic prediction, and technical regulation retrieval. Built with a focus on high-frequency data processing and stochastic modeling.

## 🚀 Key Features

* **Autonomous RAG Engine:** Utilizes **LangChain** and **ChromaDB** to vectorize 400+ technical F1 regulations, enabling context-aware retrieval for 2026 technical directives (Active Aero, X-mode/Z-mode).
* **Real-time Telemetry Pipeline:** Ingests live race data (positions, intervals, stints) via the **OpenF1 API** with a latency of <5s.
* **Stochastic Strategy Simulator:** A **Monte Carlo simulation** engine (1000+ iterations) that models overtake probabilities by accounting for "Black Swan" events like Safety Cars and mechanical DNFs.
* **Performance Analytics:** Deterministic tools for calculating windowed **Pace Deltas** and closing rates between drivers to predict DRS overtake windows.

<img width="813" height="685" alt="Screenshot 2026-03-30 at 3 08 49 AM" src="https://github.com/user-attachments/assets/f9619dba-76ce-4d29-9993-8ad1132ebfbf" />


## 🛠️ Tech Stack

* **AI/LLM:** Google Gemini 3.1 Flash, LangChain (Agentic Workflow)
* **Data Science:** NumPy (Stochastic Modeling), Pandas, Matplotlib
* **Databases:** ChromaDB (Vector Store), FastF1 (Historical Cache)
* **APIs:** OpenF1 (Live Telemetry), FastF1 (Historical Data)

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/soumil-juneja/F1-AI-race-engineer.git](https://github.com/soumil-juneja/F1-AI-race-engineer.git)
   cd F1-AI-race-engineer
