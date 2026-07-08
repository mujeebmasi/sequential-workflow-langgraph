import os
from typing import TypedDict
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv()
llm = ChatGroq(model = "llama-3.3-70b-versatile", temperature=0.7)


#always create STATE first!
#state
 

class pipeline_state(TypedDict):
    raw_input : str
    edited_text : str
    scriptwriter_text : str
    final_output : str
    


#NODES
def editor_node(state: pipeline_state) -> dict:
    """
    Stage 1: Cleans up grammer, removes typos, and refines the tone
    """
    prompt = (
    "You are an expert copyeditor. Clean up the following raw text. "
    "Fix any grammatical errors, spelling mistakes, and smooth out the transitions "
    "while keeping the core message intact. Return only the edited text.\n\n"
    f"Text:\n{state['raw_input']}"
            )
    
    response = llm.invoke(prompt)
    return {"edited_text": response.content.strip()}

def scriptwriter_node(state: pipeline_state) -> dict:
    """Stage 2: Formats and clean text into engaging video script style"""
    prompt = (
    "You are a charismatic YouTube content creator. Take this edited text and transform "
    "it into a highly engaging, punchy, conversational video script hook. Make it sound "
    "like a real person speaking passionately. Return only the script content.\n\n"
    f"Edited Text:\n{state['edited_text']}"
    
    )
    response = llm.invoke(prompt)
    return {"scriptwriter_text": response.content.strip()}

def translator_node(state: pipeline_state) -> dict:
    """Stage 3: Translates the script into natural flowing Hinglish."""
    print("\n─── [Stage 3] Executing Hinglish Translator Node ───")

    prompt = (
        "You are an expert content localizer for the Indian market. Take the following script "
        "and convert it into natural, flowing 'Hinglish'. Do not simply translate it sentence-by-sentence "
        "or repeat information. Alternating comfortably between Hindi and English phrases just like "
        "an intellectual tech educator would speak naturally on a live stream. Keep the energy high. "
        "Return only the final Hinglish text.\n\n"
        f"Script:\n{state['scriptwriter_text']}"
    )

    response = llm.invoke(prompt)
    return {"final_output": response.content.strip()}

graph = StateGraph(pipeline_state, name="Text Editing and Translation Pipeline")
graph.add_node("editor",editor_node)
graph.add_node("scriptwriter",scriptwriter_node)
graph.add_node("translator",translator_node)


#states and nodes are ready. now its time to make the GRAPH 
#for creating a graph we need to connect these nodes using edges.
#The edges define the flow of data between nodes. In this case, we will connect the nodes in a linear fashion,
#where the output of one node becomes the input for the next node.

#EDGES ARE IMPORTANT TO CREATE THE WORKFLOWS!

graph.add_edge(START, "editor")
graph.add_edge("editor", "scriptwriter")
graph.add_edge("scriptwriter", "translator")
graph.add_edge("translator", END)