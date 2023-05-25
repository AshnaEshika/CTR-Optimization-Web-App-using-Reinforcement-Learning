import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random

custom_css = """
    <style>
    .stApp {
        background-color: #e5e5f7;
        opacity: 0.8;
        background-size: 20px 20px;
        background-image: repeating-linear-gradient(0deg, #45c4f7, #45c4f7 1px, #e5e5f7 1px, #e5e5f7);
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



# Function to apply Upper Confidence Bound algorithm
def upperconfidence(dataset, X):
    # Define the number of rounds and ads
    N = X
    d = 10
    
    # Initialize variables
    ads_selected = []
    numbers_of_selections = [0] * d
    sums_of_rewards = [0] * d
    total_reward = 0
    noise_factor = 0.2
    
    # Main loop
    for n in range(0, N):
        ad = 0
        max_upper_bound = 0
        for i in range(0, d):
            if numbers_of_selections[i] > 0:
                average_reward = sums_of_rewards[i] / numbers_of_selections[i]
                delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
                noise = random.uniform(-noise_factor, noise_factor)
                upper_bound = average_reward + delta_i + noise
            else:
                upper_bound = 1e400
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                ad = i
        ads_selected.append(ad)
        numbers_of_selections[ad] += 1
        reward = dataset.values[n, ad]
        sums_of_rewards[ad] += reward
        total_reward += reward

    # Create a histogram of the ad selections
    fig, ax = plt.subplots()
    ax.hist(ads_selected)
    ax.set_title('Histogram of ads selections')
    ax.set_xlabel('Ads')
    ax.set_ylabel('Number of times each ad was selected')

    # Display the plot in Streamlit
    st.pyplot(fig)

# Streamlit app
def main():
    st.title("Upper Confidence Bound (UCB) for Ad Selection")
    st.write("This app applies the UCB algorithm to select the best ad based on historical user data.")
    
    dataset = pd.read_csv('D:/Symposium/Maximizing-Clickthrough-rate-of-ads-using-an-ensemble-approach-Thompson-Sampling-UCB--main/dataset.csv')
    
    X = st.number_input("Enter the number of rows of the dataset you want to use:", min_value=1, max_value=len(dataset))
    X = int(X)
    
    if st.button("Apply Upper Confidence Bound"):
        upperconfidence(dataset, X)
    
    st.header("Advantages of Upper Confidence Bound (UCB)")
    st.markdown("""
        - Provides a balance between exploration and exploitation, allowing the algorithm to try new ads while exploiting the best-performing ads.
        - Adapts to changes in the reward distribution over time, allowing it to quickly adjust to new trends or patterns.
        - Guarantees logarithmic regret, meaning that the cumulative regret of the algorithm grows at most logarithmically with the number of rounds.
    """)
    
    st.header("Disadvantages of Upper Confidence Bound (UCB)")
    st.markdown("""
        - Requires knowledge of the upper bound on the rewards, which may not always be available or easy to estimate accurately.
        - Can be sensitive to noise in the reward observations, leading to suboptimal decisions.
        - May require a larger number of rounds to converge to the optimal solution compared to other algorithms.
    """)

    st.header("Mathematical Formulation of Upper Confidence Bound (UCB)")
    st.latex(r'''
        \text{Upper Confidence Bound (UCB)} = \bar{X}_i + \sqrt{\frac{3}{2} \cdot \frac{\ln(n+1)}{N_i}}
    ''')

   

if __name__ == '__main__':
    main()
