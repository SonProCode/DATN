import numpy as np
import math
import random
import time
import os
import psutil

# Debug
print("DEBUG: Starting epsilon_greedy_fixed.py")

# File paths
INPUT_FILE = "/tmp/input_throughput_rtt.csv"
OUTPUT_FILE = "/tmp/pacing_gain.txt"
LOG_FILE = "/tmp/logReward.csv"
QTABLE_FILE = "/tmp/Q_Table.csv"

# RL parameters
EPSILON_INIT = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.95
ALPHA = 0.7
ACTIONS = {
    0: [1.25, 0.75, 1, 1, 1, 1, 1, 1],
    1: [2, 0.5, 1.5, 0.5, 2, 0.5, 1.5, 0.5],
    2: [1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5],
    3: [1.5, 0.75, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25],
    4: [1.11, 0.9, 1, 1, 1, 1, 1, 1],
}
NUM_ACTIONS = len(ACTIONS)

# Initialize Q-table
def init_q_table():
    return {i: 0.0 for i in range(NUM_ACTIONS)}

# Epsilon-greedy policy
def select_action(q_table, epsilon):
    if random.random() < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    best_value = max(q_table.values())
    best_actions = [a for a, v in q_table.items() if v == best_value]
    return random.choice(best_actions)

# Q-learning update
def update_q(q_table, action, reward):
    q_table[action] = (1 - ALPHA) * q_table[action] + ALPHA * reward

# File control
def check_handle(fPath):
    for proc in psutil.process_iter():
        try:
            for item in proc.open_files():
                if fPath == item.path:
                    return True
        except Exception:
            pass
    return False

def delete_file(fPath):
    if os.path.exists(fPath):
        while check_handle(fPath):
            time.sleep(0.001)
        os.remove(fPath)

def write_action(action):
    try:
        with open(OUTPUT_FILE, 'w') as f:
            f.write(str(action))
    except Exception as e:
        print(f"DEBUG: write error: {e}")

def read_rtt_throughput():
    avg_tp, avg_rtt = 0.0, 0.0
    count = 0
    while True:
        if os.path.exists(INPUT_FILE):
            with open(INPUT_FILE, 'r') as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        try:
                            tp_val, rtt_val = float(parts[0]), float(parts[1])
                            avg_tp += tp_val
                            avg_rtt += rtt_val
                            count += 1
                        except:
                            continue
            if count > 0:
                avg_tp /= count
                avg_rtt /= count
                if avg_rtt > 0 and avg_tp > 0:
                    reward = np.log10(avg_tp / avg_rtt)
                    return reward, avg_tp, avg_rtt
        time.sleep(1)

def log_score(step, reward, R, action, epsilon, best_action, tp, rtt):
    with open(LOG_FILE, "a") as f:
        f.write(f"Timestep: {step}, Reward: {reward}, Accumulate reward: {R}, Action: {action}, Epsilon: {epsilon}, Best Action: {best_action}, Throughput: {tp}, RTT: {rtt}\n")

def save_q_table(step, R, q_table, epsilon, best_action):
    with open(QTABLE_FILE, "a") as f:
        f.write(f"Time steps: {step}, Accumulate reward: {R}, {q_table}, Epsilon: {epsilon}, Best Action: {best_action}\n")

# --- MAIN LOOP ---
if __name__ == '__main__':
    epsilon = EPSILON_INIT
    q_table = init_q_table()
    step_count = 0
    R = 0.0
    initial_actions = list(range(NUM_ACTIONS))

    # Warm-up delay
    time.sleep(5)

    while True:
        if initial_actions:
            action = initial_actions.pop(0)
        else:
            action = select_action(q_table, epsilon)

        delete_file(INPUT_FILE)
        write_action(action)
        time.sleep(3)

        reward, tp, rtt = read_rtt_throughput()
        R += reward
        update_q(q_table, action, reward)
        best_action = max(q_table, key=q_table.get)

        print(f"Timestep:{step_count}, reward: {reward:.4f}, accumulate reward: {R:.4f}, action: {action}, epsilon: {epsilon:.4f}, best_action: {best_action}")
        log_score(step_count, reward, R, action, epsilon, best_action, tp, rtt)
        save_q_table(step_count, R, q_table, epsilon, best_action)

        step_count += 1
        if step_count > NUM_ACTIONS and epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY

    print("DEBUG: Finished. Final Q-table:")
    print(q_table)
