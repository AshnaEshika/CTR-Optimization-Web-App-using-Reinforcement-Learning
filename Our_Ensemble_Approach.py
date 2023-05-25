import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import streamlit as st

custom_css = """
    <style>
    .stApp {
     background-color: #ffffff;
opacity: 0.8;
background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #ffffff 10px ), repeating-linear-gradient( #c7f7b255, #c7f7b2 );
                                             
    }

    .centered-text {
        text-align: center;
    }

    .font-red {
        color: #FF0000;
    }

    .bordered-section {
        border: 2px solid #000000;
        padding: 10px;
    }
    </style>
"""

# Rest of the code...

# Streamlit app configuration
st.set_page_config(
    page_title="CTR Optimization using Ensemble Method",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit app styling
st.markdown(custom_css, unsafe_allow_html=True)

# Rest of the code...


# Function to calculate the CTR for each arm
def calculate_ctr(rewards):
    clicks = np.sum(rewards)
    impressions = len(rewards)
    ctr = clicks / impressions
    return ctr

# Function to initialize exploration parameters and reward estimates
def initialize(num_arms):
    exploration_counts = np.zeros(num_arms)
    reward_sums = np.zeros(num_arms)
    return exploration_counts, reward_sums

# Function to update the reward estimates for each arm
def update_reward_estimates(arm, reward, exploration_counts, reward_sums):
    exploration_counts[arm] += 1
    reward_sums[arm] += reward

# UCB Algorithm
def ucb_algorithm(num_arms, num_users, dataset):
    exploration_counts, reward_sums = initialize(num_arms)
    ctr_history = []
    ads_selected = []

    for user in range(num_users):
        arm_to_pull = 0
        max_ucb = 0

        for arm in range(num_arms):
            if exploration_counts[arm] == 0:
                arm_to_pull = arm
                break
            else:
                average_reward = reward_sums[arm] / exploration_counts[arm]
                exploration_bonus = math.sqrt(2 * math.log(user + 1) / exploration_counts[arm])
                ucb = average_reward + exploration_bonus
                if ucb > max_ucb:
                    max_ucb = ucb
                    arm_to_pull = arm

        reward = dataset[user, arm_to_pull]
        update_reward_estimates(arm_to_pull, reward, exploration_counts, reward_sums)
        ctr = calculate_ctr(reward_sums)
        ctr_history.append(ctr)
        ads_selected.append(arm_to_pull)

    return ctr_history, ads_selected

# Thompson Sampling Algorithm
def thompson_sampling_algorithm(num_arms, num_users, dataset):
    exploration_counts, reward_sums = initialize(num_arms)
    ctr_history = []
    ads_selected = []

    for user in range(num_users):
        sampled_theta = np.zeros(num_arms)

        for arm in range(num_arms):
            alpha = reward_sums[arm] + 1
            beta = exploration_counts[arm] - reward_sums[arm] + 1
            sampled_theta[arm] = np.random.beta(alpha, beta)

        arm_to_pull = np.argmax(sampled_theta)
        reward = dataset[user, arm_to_pull]
        update_reward_estimates(arm_to_pull, reward, exploration_counts, reward_sums)
        ctr = calculate_ctr(reward_sums)
        ctr_history.append(ctr)
        ads_selected.append(arm_to_pull)

    return ctr_history, ads_selected

# Ensemble Method
def ensemble_method(num_arms, num_users, dataset):
    ctr_history_ucb, ads_selected_ucb = ucb_algorithm(num_arms, num_users, dataset)
    ctr_history_ts, ads_selected_ts = thompson_sampling_algorithm(num_arms, num_users, dataset)

    # Combine the results using weighted averaging
    weights = [0.5, 0.5]  # Equal weights for UCB and Thompson Sampling
    ctr_history_ensemble = np.average([ctr_history_ucb, ctr_history_ts], weights=weights, axis=0)

    return ctr_history_ensemble, ads_selected_ucb

# Read the dataset from CSV
dataset = pd.read_csv('D:/Symposium/Maximizing-Clickthrough-rate-of-ads-using-an-ensemble-approach-Thompson-Sampling-UCB--main/dataset.csv')
dataset = dataset.values

# Extract
num_users = dataset.shape[0]
num_arms = dataset.shape[1]
ctr_history_ensemble, ads_selected = ensemble_method(num_arms, num_users, dataset)

# Streamlit app
st.title("CTR Optimization using Ensemble Method")
st.header("Dataset Information")
st.write("Number of users:", num_users)
st.write("Number of arms:", num_arms)

# Plotting Ads Selections Histogram
fig, ax = plt.subplots()
ax.hist(ads_selected)
ax.set_title('Histogram of Ads Selections')
ax.set_xlabel('Ads')
ax.set_ylabel('Number of times each ad was selected')

st.header("Ads Selections")
st.pyplot(fig)

# Display CTR history
st.header("CTR History")
st.line_chart(ctr_history_ensemble)

# Explanation
st.header("Explanation")
st.write("The ensemble method combines the results of the UCB (Upper Confidence Bound) algorithm and Thompson Sampling algorithm to optimize the Click-Through Rate (CTR) for different ads. It leverages the strengths of both algorithms to achieve better performance.")
st.subheader("Advantages of Ensemble Method over Thompson Sampling and UCB:")
st.markdown("""
- **Improved Robustness**: The ensemble method mitigates the weaknesses of individual algorithms by leveraging their combined strengths. It provides a more robust and reliable solution.
- **Adaptability**: The ensemble method can adapt to different scenarios and underlying reward distributions, making it more versatile and flexible.
- **Better Exploration-Exploitation Balance**: By combining UCB and Thompson Sampling, the ensemble method achieves a better trade-off between exploration (trying new arms) and exploitation (selecting arms with higher rewards).
""")
st.subheader("Mathematical Formulations:")
st.markdown("""
- **UCB Algorithm**: The Upper Confidence Bound (UCB) algorithm balances exploration and exploitation using the UCB formula for arm selection:

    `ucb = average_reward + exploration_bonus`
    
    - `average_reward` is the average reward obtained from an arm.
    - `exploration_bonus` is calculated as `sqrt(2 * log(user + 1) / exploration_counts[arm])`, where `exploration_counts[arm]` is the number of times the arm has been explored.
    
- **Thompson Sampling Algorithm**: The Thompson Sampling algorithm leverages Bayesian inference to update the reward estimates for each arm. It samples from the posterior distribution of the arm's reward probability and selects the arm with the highest sampled value.

    - `alpha` is the number of rewards obtained from the arm.
    - `beta` is the number of non-rewards obtained from the arm.
    - The reward probability of an arm is modeled as a Beta distribution: `sampled_theta[arm] = np.random.beta(alpha, beta)`.
    
- **Ensemble Method**: The ensemble method combines the results of UCB and Thompson Sampling using weighted averaging:

    `ctr_history_ensemble = np.average([ctr_history_ucb, ctr_history_ts], weights=[0.5, 0.5], axis=0)`
    
    - The weights `[0.5, 0.5]` indicate equal importance given to UCB and Thompson Sampling.
""")

