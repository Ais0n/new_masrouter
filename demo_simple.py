"""
Simple Demo for MasRouter - Shows Router Decisions
This script demonstrates what the router selects without executing the full MAS.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import torch
from loguru import logger

from MAR.MasRouter.mas_router import MasRouter
from MAR.LLM.llm_profile import llm_profile
from MAR.Agent.reasoning_profile import reasoning_profile
from MAR.Prompts.tasks_profile import tasks_profile
from MAR.Utils.utils import fix_random_seed

# Suppress verbose logging
logger.remove()
logger.add(sys.stderr, level="WARNING")

def print_box(text, width=80):
    """Print text in a box"""
    print("┌" + "─" * (width - 2) + "┐")
    for line in text.split('\n'):
        padding = width - len(line) - 4
        print(f"│ {line}{' ' * padding} │")
    print("└" + "─" * (width - 2) + "┘")

def main():
    # Example queries
    example_queries = [
        "Write a Python function to check if a number is prime.",
        "What is 15% of 240 plus 30?",
        "Why do birds fly south for the winter?",
        "Solve the equation: 3x + 7 = 22",
        "Create a function that reverses a string without using built-in reverse methods."
    ]
    
    print("\n" + "="*80)
    print("🤖 MasRouter Decision Viewer".center(80))
    print("="*80)
    print("\nThis demo shows what MasRouter SELECTS (without actually running the agents)")
    print("to save time and API costs. It demonstrates the routing decisions.\n")
    
    # Initialize
    fix_random_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_path = "mbpp_router_epoch0_new.pth"
    router = MasRouter(max_agent=6, device=device).to(device)
    
    if os.path.exists(model_path):
        print(f"✅ Loaded pre-trained model: {model_path}\n")
        router.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        router.eval()
    else:
        print(f"⚠️  No pre-trained model found. Using random initialization.\n")
    
    print("="*80)
    print("EXAMPLE QUERIES".center(80))
    print("="*80)
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n[{i}] {query}")
    
    print("\n" + "="*80 + "\n")
    
    # Get user input
    while True:
        choice = input("Enter query number (1-5) or type your own query (or 'q' to quit): ").strip()
        
        if choice.lower() in ['q', 'quit', 'exit']:
            print("\n👋 Goodbye!")
            break
        
        # Determine the query
        if choice.isdigit() and 1 <= int(choice) <= len(example_queries):
            query = example_queries[int(choice) - 1]
        elif choice:
            query = choice
        else:
            continue
        
        print("\n" + "="*80)
        print_box(f"Query: {query}")
        print("="*80)
        
        # Get embeddings only (no agent execution)
        with torch.no_grad():
            # Preprocess
            tasks_list = [f"{t['Name']} : {t['Description']}" for t in tasks_profile]
            llms_list = [f"{l['Name']} : {l['Description']}" for l in llm_profile]
            collabs_list = [f"{r['Name']} : {r['Description']}" for r in reasoning_profile]
            task_role_database, task_role_emb = router.encoder_roles()
            
            # Get embeddings
            queries_embedding = router.text_encoder([query])
            tasks_embedding = router.text_encoder(tasks_list)
            llms_embedding = router.text_encoder(llms_list)
            collabs_embedding = router.text_encoder(collabs_list)
            
            # Task classification
            selected_tasks_idx, tasks_probs, query_context = router.task_classifier(
                queries_embedding, tasks_embedding
            )
            selected_task = tasks_profile[selected_tasks_idx[0].item()]
            
            # Collaboration selection
            selected_collabs_idx, collab_log_probs, collab_context, collab_vae_loss = router.collab_determiner(
                collabs_embedding, queries_embedding
            )
            selected_collab = reasoning_profile[selected_collabs_idx[0].item()]
            
            # Number of agents
            agent_num_int, agent_num_float, num_vae_loss = router.num_determiner(queries_embedding)
            num_agents = agent_num_int[0].item()
            
            # Role selection
            tasks_role_list = [task_role_database[selected_task['Name']]]
            tasks_role_emb_list = [task_role_emb[selected_task['Name']]]
            
            selected_roles_idx, role_log_probs, role_context, role_vae_loss = router.role_allocation(
                tasks_role_emb_list,
                torch.concat([query_context, collab_context], dim=-1),
                agent_num_int
            )
            selected_roles = [tasks_role_list[0][idx.item()] for idx in selected_roles_idx[0]]
            
            # LLM selection
            selected_llms_idx, llm_log_probs, llm_vae_loss = router.llm_router(
                llms_embedding,
                torch.concat([query_context, collab_context, role_context], dim=-1),
                agent_num_int,
                agent_num_float
            )
            selected_llms = [llm_profile[idx] for idx in selected_llms_idx[0]]
        
        # Display results
        print(f"\n🎯 ROUTER DECISIONS:")
        print(f"{'─'*80}")
        print(f"📋 Task Domain:        {selected_task['Name']}")
        print(f"🤝 Collaboration:      {selected_collab['Name']}")
        print(f"👥 Number of Agents:   {num_agents}")
        print(f"\n👤 Selected Roles:")
        for i, role in enumerate(selected_roles, 1):
            print(f"   Agent {i}: {role['Name']}")
        print(f"\n🧠 Selected LLMs:")
        for i, llm in enumerate(selected_llms, 1):
            print(f"   Agent {i}: {llm['Name']}")
        print(f"{'─'*80}")
        
        # Show task probabilities
        print(f"\n📊 Task Classification Confidence:")
        for i, task in enumerate(tasks_profile):
            prob = tasks_probs[0, i].item() * 100
            bar = "█" * int(prob / 2)
            print(f"   {task['Name']:15s} [{prob:5.1f}%] {bar}")
        
        print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()
