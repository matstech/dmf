"""
DMF + LiteLLM Technical Agent Example (File/JSONL LTM Backend)

This example demonstrates how to wire up the Deterministic Memory Framework
with an LLM agent using the append-only `FileLTMHook` (JSONL) backend.
This backend acts as a write-only audit trail and does not perform active
semantic recall from LTM, relying entirely on the TemporalMemory window.
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
    # Note that in this folder's dmf_settings.toml, storage_type is set to "file"
    config_path = Path(__file__).parent / "dmf_settings.toml"
    config = load_dmf_config(config_path)

    # 2. Initialise the DMF components
    pipeline = InteractionPipeline.from_dmf_config(config)
    temporal_memory = TemporalMemory.from_dmf_config(config)
    memory = Memory.from_dmf_config(config, temporal_memory, pipeline._embedding_engine)

    # A slightly different persona for this example
    conversation_history = [
        {
            "role": "system",
            "content": (
                "You are a technical AI assistant that specializes in software engineering. "
                "You maintain context strictly using the provided memory."
            )
        }
    ]

    print("==========================================================")
    print(" Technical Agent initialized with DMF (JSONL File Backend)")
    print(" Note: Evicted memories will be archived to JSONL and not")
    print(" actively recalled from long-term memory.")
    print(" Type 'exit' or 'quit' to close the conversation.")
    print("==========================================================\n")

    while True:
        try:
            user_input = input("User: ")
        except (EOFError, KeyboardInterrupt):
            break
            
        if user_input.strip().lower() in ["exit", "quit"]:
            break
            
        # Analyze user input and add to active memory
        report, vector = pipeline.analyze_interaction_with_vector(user_input, is_system=False)
        temporal_memory.add_interaction(user_input, report, vector)
        
        # Retrieve relevant memory context from the active temporal window
        # (Since FileLTMHook is used, this won't pull old evicted records from a vector DB)
        context_str = memory.render_context(user_input)
        
        # Prepare the prompt
        messages = list(conversation_history)
        if context_str:
            messages.append({
                "role": "system",
                "content": f"Relevant active memory context:\n{context_str}"
            })
            
        messages.append({"role": "user", "content": user_input})
        
        # Generate response via LiteLLM
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
        
        # Analyze and ingest assistant reply
        ast_report, ast_vector = pipeline.analyze_interaction_with_vector(assistant_reply)
        temporal_memory.add_interaction(assistant_reply, ast_report, ast_vector)
        
        # Append to the history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": assistant_reply})

if __name__ == "__main__":
    main()
