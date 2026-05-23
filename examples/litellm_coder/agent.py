"""
DMF + LiteLLM Coding Agent Example (ChromaDB Backend)

This example demonstrates a specialized "Coding Agent" persona powered by LiteLLM
and the Deterministic Memory Framework. It shows how the LLM can leverage DMF
to remember user preferences (e.g., framework versions, coding styles) across
a session without filling up the prompt with unnecessary past chatter.
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
    pipeline = InteractionPipeline.from_dmf_config(config)
    temporal_memory = TemporalMemory.from_dmf_config(config)
    memory = Memory.from_dmf_config(config, temporal_memory, pipeline._embedding_engine)

    # 3. Define the specialized Coding Assistant system prompt
    conversation_history = [
        {
            "role": "system",
            "content": (
                "You are an expert pair-programming AI assistant. "
                "You write clean, documented, and production-ready code. "
                "The system will provide you with relevant 'temporal memory context' "
                "from past interactions with the user. Pay strict attention to any "
                "coding preferences, framework versions, or architectural decisions "
                "recorded in the memory block. If a memory says the user prefers "
                "TypeHints or a specific library, apply it automatically."
            )
        }
    ]

    print("==========================================================")
    print(" 💻 Coding Agent initialized with DMF")
    print(" Try telling it your preferences (e.g., 'I always use pytest')")
    print(" and watch how it recalls them in later questions!")
    print(" Type 'exit' or 'quit' to close the conversation.")
    print("==========================================================\n")

    while True:
        try:
            user_input = input("User: ")
        except (EOFError, KeyboardInterrupt):
            break
            
        if user_input.strip().lower() in ["exit", "quit"]:
            break
            
        # A. Analyze user input and add to active memory
        report, vector = pipeline.analyze_interaction_with_vector(user_input, is_system=False)
        temporal_memory.add_interaction(user_input, report, vector)
        
        # B. Retrieve relevant coding context (e.g., preferred frameworks, previous errors)
        context_str = memory.render_context(user_input)
        
        # C. Prepare the LLM prompt with the injected context block
        messages = list(conversation_history)
        if context_str:
            messages.append({
                "role": "system",
                "content": f"Relevant developer context from memory:\n{context_str}"
            })
            
        messages.append({"role": "user", "content": user_input})
        
        # D. Generate response via LiteLLM (using a strong coding model)
        try:
            response = litellm.completion(
                model="gpt-4o",  # Using a stronger model for coding
                messages=messages
            )
            assistant_reply = response.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            continue

        print(f"\nAssistant:\n{assistant_reply}\n")
        print("-" * 50)
        
        # E. Analyze the assistant's code/reply and ingest it into DMF
        ast_report, ast_vector = pipeline.analyze_interaction_with_vector(assistant_reply)
        temporal_memory.add_interaction(assistant_reply, ast_report, ast_vector)
        
        # F. Keep LLM sliding window clean (just recent turns, DMF handles long-term)
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": assistant_reply})
        
        # Optional: truncate conversation history to prevent infinite growth
        # since DMF is taking care of the context!
        if len(conversation_history) > 11: # 1 sys + 5 turns (user/ast)
            # keep system prompt, drop oldest turn
            conversation_history = [conversation_history[0]] + conversation_history[-10:]

if __name__ == "__main__":
    main()
