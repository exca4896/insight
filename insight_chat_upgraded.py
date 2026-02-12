import streamlit as st
import random
import time
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import shutil
import argparse
import uuid
from insight_analytics_v2_copy import QueryAnalytics
import json

load_dotenv()
CHROMA_PATH = "chroma3"

PROMPT_TEMPLATE = """
You are a top investment research AI assistant that functions as a LLM to answer client questions based on the firm's research publications.

CRITICAL CITATION REQUIREMENTS:
1. You MUST start your response by listing ALL sources with their complete metadata in this EXACT format:

**Sources:**
- Title: [exact title from document metadata]
  Published: [exact date from document metadata]
  URL: [exact URL from document metadata]
  Tag: [exact tag from document metadata]

2. After listing sources, provide your analysis.

3. When referencing information in your analysis, cite which source it came from (e.g., "According to [Title]..." or 
"As stated in the [Date] report...")

ANALYSIS GUIDELINES:
- Use ONLY the information in the provided context.
- Do NOT use prior knowledge. 
- Minimize inferring new knowledge from the provided context.
- If the answer is not explicitly contained in the context, do not infer an answer and instead direct the user to the relevant investment specialist teams [exact names from specialist_investment_managers] with their emails [exact emails from specialist_investment_managers] and assist to formulate a targeted question for the investment specialist teams in highly concise email format. If there are multiple relevant investment special teams, just list the multiple teams.
- Provide the response in detail backed by relevant information, evidence and data. 
- Organize the information and at the end section summarize the key points like a first class investment analyst. 
Include bold summary of each section so its easy to read.

IMPORTANT: The context below contains documents with metadata headers. Each document starts with [DOCUMENT N] followed by Title, Published Date, and Source URL. You MUST extract and cite these in your response.

Context:
{context}

----

Question: {question}

REMEMBER: Start your response with the **Sources:** section listing all metadata, then provide your analysis.
"""

class InsightChat:
    def __init__(self):
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

        self.analytics = QueryAnalytics()


    def load_json_data(self, json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data
        except Exception:
            return None

    def _is_unknown_response(self, response_text: str) -> bool:
        """
        Check if the AI response indicates it doesn't know the answer.
        
        Args:
            response_text: The AI's response text
            
        Returns:
            True if the response indicates the AI doesn't know, False otherwise
        """
        # Common phrases that indicate the AI doesn't know
        unknown_phrases = [
            "i don't know",
            "i do not know",
            "don't know",
            "do not know",
            "cannot find",
            "can't find",
            "not explicitly contained",
            "not found in the context",
            "based on the provided context",
            "no information",
            "unable to find",
            "not available in the context"
        ]
        
        response_lower = response_text.lower()
        
        # Check if any unknown phrase is in the response
        for phrase in unknown_phrases:
            if phrase in response_lower:
                return True
        
        return False

    def response_generator(self, query_text):
        """
        Generate response and properly log whether answer was found.
        """
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        results = db.similarity_search_with_relevance_scores(query_text, k=5)

        specialist = self.load_json_data("franklin_investment_specialist.json")


        # Case 1: No relevant documents found (low relevance score)
        if len(results) == 0 or results[0][1] < 0.7:
            self.analytics.log_query(
                query_text=query_text,
                answer_found=False,
                relevance_score=results[0][1] if results else 0.0,
                sources=[],
                response_length=0,
                session_id=st.session_state.session_id
            )
            print(f"Option 1")
            return "Unable to find matching results based on the provided context."
        
        # Case 2: Relevant documents found, ask AI
        # Build context with metadata included for each document
        context_parts = []
        json_data = self.load_json_data("franklin_templeton_insights.json")

        for idx, (doc, score) in enumerate(results, 1):
            # Create metadata header for each document chunk
            metadata_lines = [f"[DOCUMENT {idx}]"]
            
            if doc.metadata.get('title'):
                metadata_lines.append(f"Title: {doc.metadata['title']}")
            if doc.metadata.get('published_date'):
                metadata_lines.append(f"Published Date: {doc.metadata['published_date']}")
            if doc.metadata.get('source_url'):
                metadata_lines.append(f"Source URL: {doc.metadata['source_url']}")
                url = doc.metadata.get('source_url')
                if json_data != None:
                    for data in json_data:
                        print(data)
                        if url == data["url"]:
                            print(data)
                            tag = data["tags"][1]
                            metadata_lines.append(f"Tag: {tag}")

            metadata_header = "\n".join(metadata_lines)
            
            # Combine metadata header with content
            context_parts.append(f"{metadata_header}\n\nContent:\n{doc.page_content}\n\n{specialist}")
        
        # Join all document parts with clear separators
        context_text = "\n\n" + ("="*80 + "\n\n").join(context_parts)
        
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        # Optional: Print prompt for debugging (uncomment to see what's sent to LLM)
        print("="*100)
        print("PROMPT SENT TO LLM:")
        print("="*100)
        print(prompt)
        print("="*100)

        model = ChatOpenAI(model="gpt-4o", temperature=0)  # Using GPT-4o for better instruction following
        response = model.invoke(prompt)
        answer_text = response.content.strip()

        sources = [doc.metadata.get("source", "Unknown") for doc, _score in results]
        
        # Check if AI actually found an answer in the context
        answer_found = not self._is_unknown_response(answer_text)
        if answer_found == False:
            print(f"Option 2")


        # Log the query with correct answer_found status
        self.analytics.log_query(
            query_text=query_text,
            answer_found=answer_found,
            relevance_score=results[0][1],
            sources=sources if answer_found else [],
            response_length=len(answer_text),
            session_id=st.session_state.session_id
        )
        
        formatted_response = answer_text        
        return formatted_response

    def main(self):
        st.title("💬 Insight Chat")
        
        # Add analytics info in sidebar
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 📊 Analytics")
            st.markdown("""
            Track query analytics in the dashboard.
            
            Run: `streamlit run analytics_dashboard.py`
            """)
            
            st.markdown("---")
            st.markdown("### ℹ️ Session Info")
            st.caption(f"Session ID: {st.session_state.session_id[:8]}...")
            st.caption(f"Queries this session: {len(st.session_state.messages) // 2}")

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask a question about Franklin Templeton insights..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)


            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                with st.spinner("Searching insights..."):
                    response = self.response_generator(prompt)
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    st.set_page_config(
        page_title="Insight Chat",
        page_icon="💬",
        layout="wide"
    )
    
    chat = InsightChat()
    chat.main()