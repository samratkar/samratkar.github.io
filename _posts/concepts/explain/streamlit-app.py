import streamlit as st

# Set up the app title
st.title("e-x-p-l-a-i-n")

# Initialize session state for history if not already done
if 'history' not in st.session_state:
    st.session_state.history = []  # list to store Q&A pairs

# Display the Q&A history in chronological order (oldest at the top)
st.write("### Chat History")
for qa in st.session_state.history:
    msg1 = st.chat_message("user")
    msg1.write(f"**Q:** {qa['Query']}")
    msg2 = st.chat_message("assistant")
    msg2.write(f"**A:** {qa['Answer']}")
    #st.write("---")  # Divider between Q&A pairs

# Input field at the bottom of the chat history
query = st.chat_input("Say something")

# Append the Q&A to history if a question is entered
if query:
    answer = f"You typed: {query} \n"  # Replace with any answer-generating logic you like
    st.session_state.history.append({"Query": query, "Answer": answer})
    msg1 = st.chat_message("user")
    msg1.write(f"**Q:** {query}")
    msg2 = st.chat_message("assistant")
    msg2.write(f"**A:** {answer}")
