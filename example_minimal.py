"""
Minimal Example: How to Use MasRouter

This is the simplest possible example showing how to use MasRouter
with your own query.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import torch
from MAR.MasRouter.mas_router import MasRouter
from MAR.LLM.llm_profile import llm_profile
from MAR.Agent.reasoning_profile import reasoning_profile
from MAR.Prompts.tasks_profile import tasks_profile

# Suppress verbose output
import logging
logging.getLogger().setLevel(logging.ERROR)

def run_query(query_text):
    """
    Run a single query through MasRouter
    
    Args:
        query_text: Your question or prompt (string)
    
    Returns:
        tuple: (answer, cost, num_agents_used)
    """
    # Initialize the router
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    router = MasRouter(max_agent=6, device=device).to(device)
    
    # Load pre-trained model (optional but recommended)
    model_path = "mbpp_router_epoch0_new.pth"
    if os.path.exists(model_path):
        router.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    router.eval()
    
    # Run the query
    with torch.no_grad():
        results, costs, _, _, _, agent_nums = router.forward(
            queries=[query_text],
            tasks=tasks_profile,
            llms=llm_profile,
            collabs=reasoning_profile,
            given_task=None,  # Let router auto-detect task type
            prompt_file='MAR/Roles/FinalNode/gsm8k.json'
        )
    
    return results[0], costs[0], agent_nums[0].item()


if __name__ == '__main__':
    # Example 1: Math problem
    print("="*80)
    print("Example 1: Math Problem")
    print("="*80)
    query1 = "What is 25% of 80 plus 15?"
    answer1, cost1, agents1 = run_query(query1)
    print(f"Query: {query1}")
    print(f"Answer: {answer1}")
    print(f"Cost: ${cost1:.4f}")
    print(f"Agents Used: {agents1:.1f}")
    print()
    
    # Example 2: Coding problem
    print("="*80)
    print("Example 2: Coding Problem")
    print("="*80)
    query2 = "Write a Python function to check if a number is prime."
    answer2, cost2, agents2 = run_query(query2)
    print(f"Query: {query2}")
    print(f"Answer: {answer2}")
    print(f"Cost: ${cost2:.4f}")
    print(f"Agents Used: {agents2:.1f}")
    print()
    
    # Example 3: Your own query
    print("="*80)
    print("Example 3: Your Own Query")
    print("="*80)
    your_query = input("Enter your question: ")
    if your_query.strip():
        answer, cost, agents = run_query(your_query)
        print(f"\nQuery: {your_query}")
        print(f"Answer: {answer}")
        print(f"Cost: ${cost:.4f}")
        print(f"Agents Used: {agents:.1f}")
