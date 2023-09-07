import pandas as pd 
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
df=pd.read_csv("divorce_data.csv" , delimiter=";")

corr = df.corr()['Divorce'].drop('Divorce')
sort_corr=corr.abs().sort_values(ascending=False)[:20]
y = df['Divorce']
X = df.drop('Divorce',axis=1)


best=['Q18', 'Q20', 'Q40', 'Q17', 'Q16', 'Q11', 'Q19', 'Q9', 'Q25',
       'Q15', 'Q30', 'Q27', 'Q39', 'Q14', 'Q26']
X_train ,X_test , y_train , y_test = train_test_split(X[best],y , random_state=33,shuffle=True , test_size=0.1)


model=LogisticRegression(max_iter=50, penalty='l1', solver='saga')
model.fit(X_train,y_train)


# Add input widgets for user to input feature values
st.sidebar.header('Enter Information:')



# Define the questions
questions = [
    "1. My spouse and I have similar ideas about how marriage should be",
    "2. My spouse and I have similar values in trust",
    "3. We're just starting a discussion before I know what's going on.",
    "4.We share the same views about being happy in our life with my spouse ",
    "5. We're compatible with my spouse about what love should be.",
    "6. I think that one day in the future, when I look back, I see that my spouse and I have been in harmony with each other.",
    "7.My spouse and I have similar ideas about how roles should be in marriage ",
    "8. I enjoy traveling with my wife.",
    "9.I have knowledge of my spouse's inner world. ",
    "10. Our dreams with my spouse are similar and harmonious.",
    "11. I know my spouse's friends and their social relationships.",
    "12. I know what my spouse's current sources of stress are.",
    "13. Our discussions often occur suddenly.",
    "14. Most of our goals for people (children, friends, etc.) are the same",
    "15. I know my spouse's basic anxieties.",
]

# Create a Streamlit web app
st.title("Divorce Prediction App")
st.text("by Karam Alhanatleh...")



# Create input fields for user responses
user_responses = []
for i, question in enumerate(questions):
    response = st.sidebar.slider(f"Rate {question} (0-4)", 0, 4, 2)
    user_responses.append(response)

if st.sidebar.button("Predict Divorce"):
    # Make a prediction using the loaded model
    input_data = np.array(user_responses).reshape(1, -1)
    prediction = model.predict(input_data)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.write("Based on your responses, it seems unlikely that you will get a divorce.")
    else:
        st.write("Based on your responses, it seems likely that you will get a divorce.")

# Optionally, provide some additional information or instructions to the user
st.sidebar.markdown("This app predicts the likelihood of divorce based on your responses to 15 questions.")


