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

#we just made functions , now we will convert them into nodes and add them to the graph.
graph = StateGraph(pipeline_state, name="Text Editing and Translation Pipeline")
graph.add_node("editor",editor_node)
graph.add_node("scriptwriter",scriptwriter_node)
graph.add_node("translator",translator_node)


#states and nodes are ready. now its time to make the GRAPH 
#for creating a graph we need to connect these nodes using edges.
#The edges define the flow of data between nodes. In this case, we will connect the nodes in a linear fashion,
#where the output of one node becomes the input for the next node.

#EDGES ARE IMPORTANT TO CREATE THE WORKFLOWS!

# this is how we make sequential flows 
#start -> editor -> scriptwriter -> translator -> end

graph.add_edge(START, "editor")
graph.add_edge("editor", "scriptwriter")
graph.add_edge("scriptwriter", "translator")
graph.add_edge("translator", END)

#now compile the graph

app = graph.compile()
result = app.invoke(
    {
        "raw_input": """
today we are going to talk about artificial intelligence and why everybody suddenly is talking about it everywhere. many people think ai is going to take all jobs immediately but that is not actually true because ai is just a tool and like every tool it depends on who is using it and how they are using it.
if you are student or software developer or even content creator then ignoring ai right now is probably one of the biggest mistake you can make because companies are already using ai for writing code, making reports, customer support and many more things. this doesn't mean humans are useless now, instead people who knows how to use ai will replace people who don't.
the problem is most beginners keep watching tutorials after tutorials without building anything. they feel like they are learning but after one month they still cannot make a simple project from scratch. this is because learning without practicing is almost waste of time.
instead pick one project that actually solves a real problem. it doesn't need to be perfect. maybe create chatbot, resume analyzer, expense tracker or ai email assistant. while building it you will face bugs and errors and those mistakes will teach you much more than another ten hour tutorial.
so stop waiting for perfect time because it never comes. start building today even if your first project looks terrible. every expert was once beginner and the only difference is they kept building consistently while everyone else was only planning.
"""
    }
)
print("\n─── [Final Output] ───\n\n")
print(result['final_output'])