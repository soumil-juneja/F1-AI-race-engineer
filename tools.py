import pandas as pd
import chromadb
import fastf1
import os
import requests
import numpy as np
import random
import re

from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

DRIVER_NAMES = {
    1: "Norris", 81: "Piastri", 3: "Verstappen", 11: "Perez", 
    44: "Hamilton", 16: "Leclerc", 63: "Russell", 33: "Antonelli",
    14: "Alonso", 18: "Stroll", 31: "Ocon", 10: "Gasly",
    23: "Albon", 55: "Sainz", 22: "Tsunoda", 40: "Lawson",
    27: "Hulkenberg", 50: "Bearman", 77: "Bottas", 24: "Zhou"
}

print("Attempting to connect to Gemini...")

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")

os.makedirs('./f1_cache', exist_ok=True)
fastf1.Cache.enable_cache('./f1_cache')

@tool
def get_knowledge(q: str) -> str:
    """Answers general F1 questions, 2026 rules (Active Aero), and terminology."""
    try:
        client = chromadb.PersistentClient(path="./f1_db")
        collection = client.get_collection(name="f1_knowledge")
        results = collection.query(query_texts=[q], n_results=1)
        return results['documents'][0][0] if results['documents'] else "Not found."
    except Exception as e:
        return (f"CRITICAL SYSTEM ERROR: The data for this specific session is missing or has not been uploaded to FastF1 yet. "
                f"Technical reason: {e}. "
                f"DIRECTIVE: You must tell the user 'I do not have the data for this specific event yet.' DO NOT guess. DO NOT hallucinate.")
    
@tool
def get_session_summary(year: int, location: str, session_type: str, query_type: str) -> str:
    """
    Fetches the static facts of a session without complex math.
    - session_type: 'R' (Race), 'Q' (Qualifying), 'S' (Sprint), 'SQ' (Sprint Shootout).
    - query_type: 'results' (final standings), 'quali_times' (Q1/Q2/Q3), or 'overtakes' (positions gained).
    """
    try:
        session = fastf1.get_session(year, location, session_type)
        session.load(laps=True, telemetry=False, weather=False)

        if query_type == 'results':
            cols = ['Abbreviation', 'ClassifiedPosition', 'TeamName', 'Status', 'Points']
            res = session.results[cols].head(10)
            return f"Top 10 Results:\n{res.to_string()}"
        
        elif query_type == 'quali_times':
            if session_type not in ['Q', 'SQ']:
                return "Error: quali_times requires session_type 'Q' or 'SQ'."
            cols = ['Abbreviation', 'Q1', 'Q2', 'Q3']
            res = session.results[cols].head(10)
            return f"Qualifying Times:\n{res.to_string()}"
        
        elif query_type == 'overtakes':
            if session_type not in ['R', 'S']:
                return "Error: Overtakes only apply to Races (R) or Sprints (S)."
            ov = session.results[['Abbreviation', 'GridPosition', 'ClassifiedPosition']].copy()
            ov['PositionsGained'] = ov['GridPosition'] - pd.to_numeric(ov['ClassifiedPosition'], errors='coerce')
            top_movers = ov.sort_values(by='PositionsGained', ascending=False).head(5)
            return f"Top 5 Overtakers:\n{top_movers.to_string()}"
        
        return "Invalid query_type. Use 'results', 'quali_times', or 'overtakes'."
    except Exception as e:
        return (f"CRITICAL SYSTEM ERROR: The data for this specific session is missing or has not been uploaded to FastF1 yet. "
                f"Technical reason: {e}. "
                f"DIRECTIVE: You must tell the user 'I do not have the data for this specific event yet.' DO NOT guess. DO NOT hallucinate.")
    
@tool
def get_driver_stint_history(year: int, location: str, session_type: str, driver: str) -> str:
    """
    Fetches a specific driver's pit stops, lap numbers, and tire compounds used.
    - driver: 3-letter abbreviation (e.g., 'HAM', 'VER', 'ANT').
    """
    try:
        session = fastf1.get_session(year, location, session_type)
        session.load(laps = True, telemetry=False, weather=False)

        dl = session.laps.pick_drivers(driver)
        if dl.empty:
            return f"SUCCESS: Session loaded, but {driver} recorded 0 laps. This means they did not participate, DNS (Did Not Start), or crashed on lap 0."
        
        stints = dl[['Stint', 'Compound', 'TyreLife']].groupby('Stint').agg({
            'Compound': 'first',
            'TyreLife': 'max'
        }).reset_index()

        
        pit_stops = len(dl.pick_box_laps(which='in'))

        return (f"Stint History for {driver} at {location} {year}:\n"
                f"Total Pit Stops: {pit_stops}\n"
                f"Tire Strategy:\n{stints.to_string(index=False)}")
    except Exception as e:
        return (f"CRITICAL SYSTEM ERROR: The data for this specific session is missing or has not been uploaded to FastF1 yet. "
                f"Technical reason: {e}. "
                f"DIRECTIVE: You must tell the user 'I do not have the data for this specific event yet.' DO NOT guess. DO NOT hallucinate.")
    
BASE_URL = "https://api.openf1.org/v1"

@tool
@tool
def get_live_summary(session_key: str = "latest") -> str:
    """
    Returns a comprehensive summary of the top 5 drivers from the active/test session.
    Includes: Position, Name, Tyre Compound, and Pit Stop Count.
    """
    # Hardcoded Bahrain 2024 (9472) for stable testing
    session_key = "9472"
    
    try:
        pos_resp = requests.get(f"{BASE_URL}/positions?session_key={session_key}").json()
        stint_resp = requests.get(f"{BASE_URL}/stints?session_key={session_key}").json()
        pit_resp = requests.get(f"{BASE_URL}/pit?session_key={session_key}").json()

        if not isinstance(pos_resp, list) or not pos_resp:
            return "TRACK STATUS: No active telemetry detected. The session may be over or the track is cold."

        latest_pos = {}
        for p in pos_resp:
            if isinstance(p, dict) and 'driver_number' in p:
                num = p['driver_number']
                if num not in latest_pos or p.get('date', '') > latest_pos[num].get('date', ''):
                    latest_pos[num] = p
        
        top_5_drivers = sorted(latest_pos.values(), key=lambda x: x.get('position', 99))[:5]
        
        summary = ["--- PIT WALL RACE SUMMARY (LIVE FEED) ---"]
        
        for driver in top_5_drivers:
            d_num = driver.get('driver_number')
            if d_num is None: continue
            
            name = DRIVER_NAMES.get(d_num, f"Driver {d_num}")
            
            current_stint = {'compound': 'UNKNOWN', 'lap_start': 0}
            if isinstance(stint_resp, list):
                d_stints = [s for s in stint_resp if isinstance(s, dict) and s.get('driver_number') == d_num]
                if d_stints:
                    current_stint = d_stints[-1]
            
    
            d_pits = 0
            if isinstance(pit_resp, list):
                d_pits = len([p for p in pit_resp if isinstance(p, dict) and p.get('driver_number') == d_num])
            
            summary.append(
                f"P{driver.get('position', '?')}: {name} | "
                f"Tyre: {current_stint.get('compound')} (Stint Start: Lap {current_stint.get('lap_start')}) | "
                f"Pits: {d_pits}"
            )
        
        return "\n".join(summary)

    except Exception as e:
        return f"CRITICAL TOOL ERROR: {e}"

def calculate_pace_delta(driver_a: int, driver_b: int, window: int = 5) -> str:
    """
    Calculates the average pace difference between two drivers over the last X laps.
    Window defaults to 5 laps to filter out 'noise' like DRS or minor mistakes.
    """
    session_key = "9472" # Hardcoded Bahrain 2024 for testing
    try:
        resp_a = requests.get(f"{BASE_URL}/laps?session_key={session_key}&driver_number={driver_a}").json()
        resp_b = requests.get(f"{BASE_URL}/laps?session_key={session_key}&driver_number={driver_b}").json()

        if not isinstance(resp_a, list) or not isinstance(resp_b, list):
            return "DATA ERROR: API returned a message instead of lap timing. The session may be inactive."

        def get_clean_laps(data):
            return [l.get('lap_duration') for l in data if isinstance(l, dict) and l.get('lap_duration') is not None]

        laps_a = get_clean_laps(resp_a)[-window:]
        laps_b = get_clean_laps(resp_b)[-window:]

        if len(laps_a) < 2 or len(laps_b) < 2:
            return f"INSUFFICIENT DATA: Not enough clean laps found for Driver {driver_a} or {driver_b}."

        avg_a = sum(laps_a) / len(laps_a)
        avg_b = sum(laps_b) / len(laps_b)
        delta = avg_a - avg_b

        return (f"--- PACE ANALYSIS (Last {window} Laps) ---\n"
                f"Driver {driver_a} Avg: {avg_a:.3f}s | Driver {driver_b} Avg: {avg_b:.3f}s\n"
                f"DELTA: Driver {driver_a} is {abs(delta):.3f}s {'faster' if delta < 0 else 'slower'} per lap.")
    except Exception as e:
        return f"Pace Analysis Failure: {e}"

@tool
def predict_overtake_window(chaser: int, leader: int) -> str:
    """
    Uses deterministic math to predict the exact lap a chaser will reach the leader.
    Formula: Current Gap / Average Closing Rate.
    """
    session_key = "9472"
    try:
        int_resp = requests.get(f"{BASE_URL}/intervals?session_key={session_key}&driver_number={chaser}").json()
        if not isinstance(int_resp, list) or not int_resp:
            return "ERROR: Could not retrieve current interval data."
        
        current_gap = int_resp[-1].get('gap_to_leader', 0) 

        pace_report = calculate_pace_delta.run(driver_a=chaser, driver_b=leader, window=3)
        if "slower" in pace_report:
            return f"OVERTAKE UNLIKELY: Chaser (Driver {chaser}) is currently slower than the leader."

        match = re.search(r"is (\d+\.\d+)s faster", pace_report)
        if not match: return "ANALYSIS FAILED: Could not establish a stable closing rate."
        
        closing_rate = float(match.group(1))
        laps_to_catch = current_gap / closing_rate

        return (f"--- DETEERMINISTIC PREDICTION ---\n"
                f"Current Gap: {current_gap}s | Closing Rate: {closing_rate:.3f}s/lap\n"
                f"ESTIMATE: Contact expected in approximately {round(laps_to_catch, 1)} laps.")
    except Exception as e:
        return f"Predictor Failure: {e}"

@tool
def simulate_strategic_chaos(chaser: int, leader: int, laps_remaining: int) -> str:
    """
    The Strategist: A Monte Carlo simulation (1000 iterations) that models 
    stochastic lap variance and 'Black Swan' events like Safety Cars and DNFs.
    """
    PROB_SAFETY_CAR = 0.015 
    PROB_DNF = 0.005        
    

    base_gap = 2.5 
    avg_closing_rate = 0.35 
    
    simulations = 1000
    overtakes, black_swans = 0, 0

    try:
        for _ in range(simulations):
            sim_gap = base_gap
            for lap in range(int(laps_remaining)):
                # 1. Check for (Mechanical Failure)
                if random.random() < PROB_DNF:
                    break 
                
                # 2. Check for Safety Car 
                if random.random() < PROB_SAFETY_CAR:
                    black_swans += 1
                    sim_gap = 0.5 
                
                # Normal Distribution 
                sim_gap -= np.random.normal(avg_closing_rate, 0.15)
                
                if sim_gap <= 0:
                    overtakes += 1
                    break

        prob = (overtakes / simulations) * 100
        return (f"--- STRATEGIC MONTE CARLO REPORT ---\n"
                f"Overtake Probability: {prob:.1f}% across {simulations} iterations.\n"
                f"Black Swan Incidents Simulated: {black_swans}\n"
                f"STRATEGY: {'Box for fresh tires' if prob < 30 else 'Maintain current stint - Overtake likely'}")
    except Exception as e:
        return f"Simulation Failure: {e}"



# --- THE REGISTRY ---
# Expose this list so app.py can import it
F1_TOOLS = [
    get_knowledge, 
    get_session_summary, 
    get_driver_stint_history,
    get_live_summary,
    calculate_pace_delta,
    simulate_strategic_chaos
]

