from utils import *

def llm_lineage(lineage_):
    if lineage_ is not None:
        li = ["Select question to get the lineage",
            "What is the customer name?",
            "What is the suspect's name?",
            "List the merchant name",
            "How was the bank notified?",
            "When was the bank notified?",
            "What type of fraud is taking place?",
            "When did the fraud occur?",
            "Was the disputed amount greater than 5000 usd?",
            "What type of network/card is used in transaction?",
            "Was the police report filed?"]
        
        
        selected_option = st.selectbox("", li)
        if selected_option in li[1:]:
            doc = lineage_[selected_option]
            for i in range(len(doc)):
                y = i+1
                st.write(f":blue[Chunk-{y}:]")
                st.write(":blue[Page Content:]", doc[i].page_content)
                st.write(":blue[Source:]",doc[i].metadata['source'])
