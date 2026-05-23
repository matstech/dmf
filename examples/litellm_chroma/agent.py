"""
DMF + LiteLLM Generic Agent Example (ChromaDB LTM Backend)

This example demonstrates how to wire up the Deterministic Memory Framework
with a generic LLM agent powered by LiteLLM. It uses the default ChromaDB
backend for Long-Term Memory (LTM).
"""

import os
from pathlib import Path

import litellm

from dmf.memory.api import Memory
from dmf.memory.temporal_memory import TemporalMemory
from dmf.runtime.pipeline import InteractionPipeline
from dmf.utils.config_loader import load_dmf_config

# Ensure you have your API key exported, e.g.:
# export OPENAI_API_KEY="sk-..."
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY not found in environment.")
    print("LiteLLM requires it to call OpenAI models.")


def main():
    # 1. Load configuration from the local dmf_settings.toml
    config_path = Path(__file__).parent / "dmf_settings.toml"
    config = load_dmf_config(config_path)

    # 2. Initialise the DMF components
    # The pipeline handles NLP extraction, embeddings, and geometry calculations
    pipeline = InteractionPipeline.from_dmf_config(config)
    
    # TemporalMemory manages the active sliding window, pruning, and LTM hooks
    temporal_memory = TemporalMemory.from_dmf_config(config)
    
    # Memory is the public facade for candidate retrieval and context rendering
    memory = Memory.from_dmf_config(config, temporal_memory, pipeline._embedding_engine)

    # Agent conversation history (for the LLM context window)
    # Note: DMF tracks its own internal state; this history is just for LiteLLM.
    conversation_history = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. Use the provided memory context "
                "to answer the user's questions accurately. If the memory provides "
                "instructions or constraints, follow them strictly."
            )
        }
    ]

    print("=====================================================")
    print(" Generic Agent initialized with DMF (ChromaDB Backend)")
    print(" Type 'exit' or 'quit' to close the conversation.")
    print("=====================================================\n")

    while True:
        try:
            user_input = input("User: ")
        except (EOFError, KeyboardInterrupt):
            break
            
        if user_input.strip().lower() in ["exit", "quit"]:
            break
            
        # A. Analyze user input and ingest it into DMF
        report, vector = pipeline.analyze_interaction_with_vector(user_input, is_system=False)
        temporal_memory.add_interaction(user_input, report, vector)
        
        # B. Retrieve relevant memory context based on the current query
        # This triggers Candidate Generation, Answerability Reranking, and Evidence Assembly
        context_str = memory.render_context(user_input)
        
        # C. Prepare the prompt for the LLM
        messages = list(conversation_history)
        if context_str:
            messages.append({
                "role": "system",
                "content": f"Relevant memory context from the past:\n{context_str}"
            })
            
        messages.append({"role": "user", "content": user_input})
        
        # D. Generate response via LiteLLM
        try:
            response = litellm.completion(
                model="gpt-4o-mini",
                messages=messages
            )
            assistant_reply = response.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            continue

        print(f"\nAssistant: {assistant_reply}\n")
        
        # E. Analyze the assistant's reply and ingest it into DMF
        # This completes the interaction loop and allows the framework to compute
        # semantic divergence and topics for both sides of the conversation.
        ast_report, ast_vector = pipeline.analyze_interaction_with_vector(assistant_reply)
        temporal_memory.add_interaction(assistant_reply, ast_report, ast_vector)
        
        # F. Append to the local LLM conversation history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": assistant_reply})

if __name__ == "__main__":
    main()
