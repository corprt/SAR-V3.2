from utils import *
from closed_source2 import *
from open_source import *
 

def decision_gpt(decision,temp_file_path,hf_embeddings):
    
    docs, docsearch = embedding_store(temp_file_path,hf_embeddings)   
      
    if decision:
         
        st.write("#### *SARA Recommendation*")
        
        query ="Is invoice is billed to cardholder or someone else?"
        contexts = docsearch.similarity_search(query, k=9) 
        prompt = f" You are professional Fraud Analyst. Find answer to the questions as truthfully and in as detailed as possible as per given context only,\n\n\
        cardholder's name,adress can be identified from cardholder information. Customer is the person who is the owner of the card, customer can also be referenced as the victim with home fraud has taken place.\n\n\
        Identify to whom invoice is billed. (Detials mentioned in invoice is of the person who made the transaction,it may be or may not be of the customer)\n\n\
        Compare both the details, if details mentioned in invoice matches the cardholder details, then invoice is billed to customer else it is billed to someone else who misued the card.\n\n\
            Context: {contexts}\n\
            Response (Give me a concise response.)"
        response_5 = usellm(prompt) 

        


        query ="Give your recommendation if SAR filling is required or not?"
        contexts = docsearch.similarity_search(query, k=9) 
        prompt = f" You are professional Fraud Analyst. Find answer to the questions as truthfully and in as detailed as possible as per given context only,\n\n\
        SAR refers to Suspicious activity Report, which is a document that financial institutions must file with the Financial Crimes Enforcement Network (FinCEN) based on the Bank Secrecy Act whenever there is a suspicious activity.\n\n\
        1. Check if The transaction/disputed amount > 5,000 USD value threshold,If Yes, then check below points to make sure if it is a suspicious activity or not: \n\
        2. {response_5} analyse this response,if details matches or not? If matches then there is no suspicion else, it can be a suspicipos activity.\n\n\
        3. {response_5} analyse this response and find if there is a misuse of the card, if Yes, then identify the person name who misused the card, If a person is identified then this can be a suspicious activity, else not.\n\n\
        If no suspicious activity is detected based on above mentioned points, write your response as - There is no indication of suspicious activity.Therefore,no requirement to file SAR with FinCEN.\n\n\
        Context: {contexts}\n\
        Response (Give me a concise response in few pointers.Kindly mention if one should file SAR with FinCEN or not)"       
        
        response_sara_gpt = usellm(prompt) 
        response_sara_gpt = response_sara_gpt.replace("$", " ")
        response_sara_gpt = response_sara_gpt.replace("5,000", "5,000 USD")
        response_sara_gpt = response_sara_gpt.replace("5,600", "5,600 USD")

        st.markdown(f'''<em>{response_sara_gpt}</em>''',unsafe_allow_html=True)

        st.warning('Please carefully review the recommendation and case details before the final submission',icon="⚠️")

   
            
def decision_llama(decision,temp_file_path,hf_embeddings):
    docs, docsearch = embedding_store(temp_file_path,hf_embeddings) 
    if decision: 

        st.write("#### *SARA Recommendation*")

        # Create a retriever from the vector store
        retriever = docsearch.as_retriever(search_kwargs={"k": 9})
        # Prompt
        template = """Act as a professional Fraud Analyst. Find answer to the questions as truthfully and in as detailed as possible as per given context only,
        Identify cardholder's name,adress from cardholder information. Customer is the person who is the owner of the card and with whom fraud has taken place.
        Identify name and address to whom merchant invoice is billed 
        Identify if cardholder details matched with billed details on Invoice that is if Invoice is billed to cardholder or someone else based on above information.
        If Invoice is billed to someone else, then that could be the potential suspect.
        {context}
        Question: {query}
        Helpful Answer:"""

        query = "If Merchant Invoice is billed to cardholder or someone else?"

        def run_chain(template,query):
        
            QA_CHAIN_PROMPT = PromptTemplate(
                input_variables=["context", "query"],
                template=template,
            )
        
            # Docs
            docs = retriever.get_relevant_documents(query)

            # Chain
            chain = load_qa_chain(llama_13b, chain_type="stuff", prompt=QA_CHAIN_PROMPT)
    
            # Run
            response = chain({"input_documents": docs, "query": query}, return_only_outputs=True)

            return response
        

        response_3 = run_chain(template,query)
        
        analyse = response_3['output_text']
    



        query = "Give your Recommendation if SAR filling is required or not?"
        contexts = docsearch.similarity_search(query, k=5)
        prompt = f'''Act as a financial analyst and give concise answer to the question, with given Context.\n\n\
        SAR refers to Suspicious activity Report, which is a document that financial institutions must file with the Financial Crimes Enforcement Network (FinCEN) based on the Bank Secrecy Act.\n\n\
        Check if The transaction/disputed amount > 5,000 USD value threshold,If Yes, then check below points to make sure if it is a suspicious activity or not: \n\
        1. {analyse} Analyse this properly,if invoice is billed to cardholder, then there is no suspicion else, it can be a suspicious activity.(Mention mismatched details)\n\n\
        2. {analyse} If a potential suspect is identified then this can be considered as a suspicious activity else not.(Mention suspect name)\n\n\
        Even if transaction/disputed amount > 5,000 USD but if above criteria does not met, then this can not be considered as a suspicious activity. \n\n\
        Analyse above points properly and give your recommendation if this is a case of suspicious activity or not? \n\n\
        Context: {contexts}\n\
        Response (Give me a concise response in 3-4 points.Kindly mention if one should file SAR with FinCEN or not)'''
                
                
        response_sara_llama = llama_llm(llama_13b,prompt)
        response_sara_llama = response_sara_llama.replace("$", "USD")
        response_sara_llama = response_sara_llama.replace("5,000", "5,000")
        response_sara_llama = response_sara_llama.replace("5,600", "5,600")

        # st.write(response_sara_llama)

        st.markdown(f'''<em>{response_sara_llama}</em>''',unsafe_allow_html=True)
        
        # st.write(f'''<em>{response_sara_llama}</em>''',unsafe_allow_html=True)

        st.warning('Please carefully review the recommendation and case details before the final submission',icon="⚠️")

def selection1(decision):  
    if decision: 
        selected_rad = st.radio(":blue", ["Yes", "No", "Refer for review"], horizontal=True,disabled=st.session_state.disabled)
        if selected_rad == "Refer for review":
            email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            email_id = st.text_input("Enter email ID")
            if email_id and not re.match(email_regex, email_id):
                st.error("Please enter a valid email ID")


        if st.button("Submit"):
            if selected_rad in ("Yes"):
                st.warning("Thanks for your review, your response has been submitted")
            elif selected_rad in ("No"):
                st.success("Thanks for your review, your response has been submitted")

            else:
                st.info("Thanks for your review, Case has been assigned to the next reviewer")


def selection2(decision):
    if decision:       
        selected_rad = st.radio(":blue", ["No", "Yes", "Refer for review"], horizontal=True,disabled=st.session_state.disabled)
        if selected_rad == "Refer for review":
            email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            email_id = st.text_input("Enter email ID")
            if email_id and not re.match(email_regex, email_id):
                st.error("Please enter a valid email ID")


        if st.button("Submit"):
            if selected_rad in ("Yes"):
                st.warning("Thanks for your review, your response has been submitted")
            elif selected_rad in ("No"):
                st.success("Thanks for your review, your response has been submitted")
            else:
                st.info("Thanks for your review, Case has been assigned to the next reviewer")





