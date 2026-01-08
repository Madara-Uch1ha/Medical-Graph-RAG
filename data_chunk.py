"""
Data chunking module - Updated for LangChain 1.x compatibility
Uses with_structured_output() instead of deprecated create_extraction_chain
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field
from typing import List
import os
from dataloader import load_high
from agentic_chunker import AgenticChunker


# Pydantic data class for structured output
class Sentences(BaseModel):
    """List of sentences/propositions extracted from text"""
    sentences: List[str] = Field(description="List of extracted propositions/sentences")


def get_propositions(text: str, llm_with_structure) -> List[str]:
    """Extract propositions from text using structured output"""
    if not text.strip():
        return []
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the key propositions and facts from the following text. Return them as a list of clear, atomic sentences."),
        ("user", "{input}")
    ])
    
    chain = prompt | llm_with_structure
    
    try:
        result = chain.invoke({"input": text})
        if hasattr(result, 'sentences'):
            return result.sentences
        return []
    except Exception as e:
        print(f"Error extracting propositions: {e}")
        return []


def run_chunk(essay: str) -> List[str]:
    """
    Process essay into semantic chunks using agentic chunking.
    
    Args:
        essay: The text to chunk
        
    Returns:
        List of semantic chunks
    """
    # Initialize LLM with structured output
    llm = ChatOpenAI(
        model='gpt-5-nano-2025-08-07', 
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )
    
    # Create LLM with structured output (replacement for create_extraction_chain)
    llm_with_structure = llm.with_structured_output(Sentences)

    paragraphs = essay.split("\n\n")
    essay_propositions = []

    for i, para in enumerate(paragraphs):
        if para.strip():
            propositions = get_propositions(para, llm_with_structure)
            essay_propositions.extend(propositions)
            print(f"Done with paragraph {i}")

    # Use agentic chunker to group propositions
    ac = AgenticChunker()
    ac.add_propositions(essay_propositions)
    ac.pretty_print_chunks()
    chunks = ac.get_chunks(get_type='list_of_strings')

    return chunks