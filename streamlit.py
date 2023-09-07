import streamlit as st

# Set the title of your app
st.title("Simple Streamlit Example")

# Add some text to your app
st.text("This is a simple Streamlit app.")

# Add a button to your app
if st.button("Click Me"):
    st.write("You clicked the button!")
