# from langchain.llms import OpenAI
from utils import *
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline




# Setting Env
if st.secrets["OPENAI_API_KEY"] is not None:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")   


text_splitter = RecursiveCharacterTextSplitter(
chunk_size = 650,
chunk_overlap  = 0,
length_function = len,
separators=[]
)


@st.cache_data(show_spinner=False)
def embedding_store(text,_hf_embeddings):
    docs = text_to_docs(text)
    docsearch = FAISS.from_documents(docs, _hf_embeddings)
    return docs,docsearch
    
# Memory setup for gpt-3.5
llm = ChatOpenAI(temperature=0.1)

def pretty_print_docs(docs):
    st.write(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def retriever1(temp_file_path,hf_embeddings):

    docs, docsearch = embedding_store(temp_file_path,hf_embeddings) 

    # Create a retriever from the vector store
    retriever = docsearch.as_retriever()

    # Create a RetrievalQA object using the LLM and the retriever
    retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=retriever)


    # Use the `apply()` or `call()` method to run the chain on a list of inputs or a single input. The inputs can be queries or prompts that you want to use to retrieve relevant documents.
    results = retrievalQA.apply(["What is the cardholder name?", "What is the fraud type?","What is the suspect name?"])

    st.write(results)
 


def retriever2(temp_file_path,hf_embeddings):
    docs, docsearch = embedding_store(temp_file_path,hf_embeddings) 

    # Create a retriever from the vector store
    retriever = docsearch.as_retriever(search_kwargs={"k": 2})

    docs = retriever.get_relevant_documents("What is the cardholder name?")
    pretty_print_docs(docs)


def retriever3(temp_file_path,hf_embeddings):
    docs, docsearch = embedding_store(temp_file_path,hf_embeddings) 

    # Create a retriever from the vector store
    retriever = docsearch.as_retriever(search_kwargs={"k": 3})

    _filter = LLMChainFilter.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=_filter, base_retriever=retriever)

    compressed_docs = compression_retriever.get_relevant_documents("What is the cardholder name?")
    st.write(compressed_docs)
    pretty_print_docs(compressed_docs)


def retriever4(temp_file_path,hf_embeddings,query1,query2):
    docs, docsearch = embedding_store(temp_file_path,hf_embeddings) 
    # Create a retriever from the vector store
    retriever = docsearch.as_retriever(search_kwargs={"k": 3})
    embeddings_filter = EmbeddingsFilter(embeddings=hf_embeddings, similarity_threshold=0.80)
    compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)
    compressed_docs = compression_retriever.get_relevant_documents(query1)
    st.write("Lineage-")
    # st.write(compressed_docs[1].page_content)
    pretty_print_docs(compressed_docs)
    # format_docs(compressed_docs)
    
    context_1 = compressed_docs[1].page_content
    prompt_1 =  f'''Identify the source,document name of the question asked that is from where the answer is identified. (Document name is the tile/first heading)\n\n\
                Question: {query2}\n\
                Context: {context_1}\n\
                Response: '''
    response = usellm(prompt_1)
    st.write("Source")
    st.write(response)
  


def retriever5(temp_file_path,hf_embeddings):
    docs, docsearch = embedding_store(temp_file_path,hf_embeddings) 
    # Create a retriever from the vector store
    retriever = docsearch.as_retriever(search_kwargs={"k": 2})
    llm = ChatOpenAI(temperature=0.1)
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    compressed_docs = compression_retriever.get_relevant_documents("What is the cardholder name?")
    st.write(compressed_docs)
    # pretty_print_docs(compressed_docs)


def retriever6(temp_file_path,hf_embeddings):
    docs, docsearch = embedding_store(temp_file_path,hf_embeddings) 
    retriever = docsearch.as_retriever(search_kwargs={"k": 1})
    redundant_filter = EmbeddingsRedundantFilter(embeddings=hf_embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=hf_embeddings, similarity_threshold=0.76)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[redundant_filter, relevant_filter]
        )
    compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)

    compressed_docs = compression_retriever.get_relevant_documents("What is the cardholder name?")
    # pretty_print_docs(compressed_docs)
    format_docs(compressed_docs)





