# 🏎️ F1 AI Race Engineer

An autonomous, agentic system designed for real-time motorsport analytics, strategic prediction, and technical regulation retrieval. Built with a focus on high-frequency data processing and stochastic modeling.

## 🚀 Key Features

* **Autonomous RAG Engine:** Utilizes **LangChain** and **ChromaDB** to vectorize 400+ technical F1 regulations, enabling context-aware retrieval for **2026 technical directives** (Active Aero, X-mode/Z-mode).
* **Real-time Telemetry Pipeline:** Ingests live race data (positions, intervals, stints) via the **OpenF1 API** with a latency of **<5s**.
* **Stochastic Strategy Simulator:** A **Monte Carlo simulation** engine (1000+ iterations) that models overtake probabilities by accounting for "Black Swan" events like Safety Cars and mechanical DNFs.
* **Performance Analytics:** Deterministic tools for calculating windowed **Pace Deltas** and closing rates between drivers to predict DRS overtake windows.

<img width="813" height="685" alt="Screenshot 2026-03-30 at 3 08 49 AM" src="https://github.com/user-attachments/assets/f9619dba-76ce-4d29-9993-8ad1132ebfbf" />

---

### 🛠️ Specialized Toolset & Engineering Logic

The system orchestrates a suite of custom-built tools designed to solve specific motorsport engineering challenges. Instead of simple text generation, the agent executes Python-based logic to provide grounded, quantitative answers.

### 🔍 `get_knowledge` (Agentic RAG)
* **Engine:** LangChain + ChromaDB (Vector Store).
* **Logic:** Performs a semantic similarity search across vectorized PDF documents of the **2026 FIA Technical Regulations**. It retrieves specific clauses on Power Unit energy recovery ($MGU-K$) and Aerodynamic "X-mode/Z-mode" to ensure strategy advice is regulation-compliant.

### 🏁 `get_session_summary` (Static Data)
* **Data Source:** FastF1 (Historical/Static).
* **Logic:** Retrieves "Cold Data" from completed sessions, including final race classifications, qualifying positions, and driver lineups. It serves as the primary ground-truth source for post-race debriefs and performance benchmarking.

### 🛞 `get_driver_stint_history` (Strategy Tracker)
* **Data Source:** FastF1 (Stint/Tire Data).
* **Logic:** Extracts high-resolution historical data on a driver's tire usage, pit stop intervals, and compound longevity. This tool allows the agent to identify "offset" strategies and predict when a driver's pace will drop due to tire degradation.

### 📡 `get_live_summary` (Real-Time Pulse)
* **Data Source:** OpenF1 API (Live).
* **Logic:** Provides a high-density "Snapshot" of the current leaderboard. It pulls real-time data on the Top 5 drivers, including current interval gaps, active tire compounds, and total pit stops made, ensuring the agent's strategy advice is synchronized with the live race state.

### ⏱️ `calculate_pace_delta` (Telemetry Analytics)
* **Engine:** OpenF1 API (REST).
* **Logic:** Ingests the last 5–10 laps of telemetry for any two specified drivers. It calculates a Moving Average Lap Time and identifies the **Closing Rate ($m/s$)** to determine if a chasing driver is genuinely faster or simply benefiting from a temporary tow.

### 🎲 `simulate_strategic_chaos` (Stochastic Simulator)
* **Engine:** NumPy / SciPy.
* **Logic:** A 1,000-iteration Monte Carlo engine. It models the remaining race distance by injecting random variables such as **Lap Time Variance** (Gaussian noise based on tire degradation) and **Event Triggers** (Probability of SC/VSC based on track history).
* **Result:** Returns a probability distribution (e.g., *"72% chance that an Undercut on Lap 18 gains net track position"*).

### 📊 `plot_quali_vs_race` (Performance Benchmarking)
* **Engine:** FastF1 + Matplotlib.
* **Logic:** Extracts high-frequency (20Hz) telemetry including Throttle, Brake, and Gear data. It overlays a driver’s best Qualifying lap against their average Race lap to visualize performance deltas during fuel-heavy stints.

---

## 📈 System Workflow (HLD)

1.  **Ingestion:** The Streamlit UI captures a strategic query (e.g., *"Can Hamilton catch Russell before the end of the stint?"*).
2.  **Reasoning:** The Gemini 3.1 Agent identifies that it needs current gaps (OpenF1) and a projected finish (Simulation).
3.  **Execution:** The agent calls `calculate_pace_delta` followed by `simulate_strategic_chaos` in a multi-step **ReAct loop**.
4.  **Synthesis:** Raw NumPy arrays and JSON telemetry are synthesized into a concise "Race Engineer" radio message for the user.

---

## 🛠️ Tech Stack & Prerequisites

* **AI/LLM:** Google Gemini 3.1 Flash, LangChain (Agentic Workflow)
* **Data Science:** NumPy (Stochastic Modeling), Pandas, Matplotlib
* **Databases:** ChromaDB (Vector Store), FastF1 (Historical Cache)
* **APIs:** OpenF1 (Live Telemetry), FastF1 (Historical Data)
* **Requirement:** Python 3.10+, Google Gemini API Key

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/soumil-juneja/F1-AI-race-engineer.git](https://github.com/soumil-juneja/F1-AI-race-engineer.git)
   cd F1-AI-race-engineer
