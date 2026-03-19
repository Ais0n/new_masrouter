"""
Interactive Demo for MasRouter Multi-Agent System
This script allows you to interact with the MasRouter system using your own prompts.
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
from MAR.Utils.globals import Cost
from MAR.Utils.log import configure_logging

# Configure logging
configure_logging(log_name="demo_interactive.txt")

def print_separator(char="=", length=80):
    print(char * length)

def print_section(title):
    print_separator()
    print(f"🔹 {title}")
    print_separator()

def display_options(title, items, show_description=True):
    """Display available options with indices"""
    print(f"\n{title}:")
    for i, item in enumerate(items):
        if show_description:
            print(f"  [{i}] {item['Name']}")
            print(f"      {item['Description'][:100]}...")
        else:
            print(f"  [{i}] {item['Name']}")
    print()

def main():
    print_separator("=")
    print("🤖 Welcome to MasRouter Interactive Demo!")
    print_separator("=")
    print("\nThis demo allows you to:")
    print("  • Enter your own queries/prompts")
    print("  • See how MasRouter automatically selects:")
    print("    - Task type (Math, Code, Commonsense)")
    print("    - Collaboration method (CoT, Debate, Reflection, etc.)")
    print("    - Number of agents needed")
    print("    - Specific roles for each agent")
    print("    - Which LLMs to use")
    print("  • View the final result from the multi-agent system")
    print_separator("=")
    
    # Initialize
    fix_random_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load pre-trained model if available
    model_path = "mbpp_router_epoch0_new.pth"
    router = MasRouter(max_agent=6, device=device).to(device)
    
    if os.path.exists(model_path):
        logger.info(f"Loading pre-trained model from {model_path}")
        router.load_state_dict(torch.load(model_path, map_location=device))
        router.eval()
    else:
        logger.warning(f"No pre-trained model found at {model_path}. Using randomly initialized model.")
    
    # Display available options
    print_section("Available Options")
    display_options("📋 Tasks", tasks_profile, show_description=False)
    display_options("🤝 Collaboration Methods", reasoning_profile, show_description=False)
    display_options("🧠 Available LLMs", llm_profile, show_description=False)
    
    # Determine prompt file based on task
    task_prompt_files = {
        'Math': 'MAR/Roles/FinalNode/gsm8k.json',
        'Code': 'MAR/Roles/FinalNode/mbpp.json',
        'Commonsense': 'MAR/Roles/FinalNode/gsm8k.json',  # fallback
    }
    
    print_section("Interactive Session")
    print("Enter your queries below. Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        try:
            # Get user input
            print_separator("-")
            user_query = input("\n💬 Your Query: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using MasRouter! Goodbye!")
                break
            
            if not user_query:
                print("⚠️  Please enter a valid query.")
                continue
            
            # Reset cost counter
            Cost.instance().reset()
            
            # Run the router
            print("\n🔄 Processing your query through MasRouter...")
            print("   (This may take a moment as it calls multiple LLMs)")
            
            with torch.no_grad():
                results, costs, log_probs, tasks_probs, vae_loss, agent_num_float = router.forward(
                    queries=[user_query],
                    tasks=tasks_profile,
                    llms=llm_profile,
                    collabs=reasoning_profile,
                    given_task=None,  # Let the router decide
                    prompt_file='MAR/Roles/FinalNode/gsm8k.json'  # Default, will be adjusted by router
                )
            
            # Display results
            print_section("Results")
            print(f"📝 Your Query: {user_query}\n")
            print(f"✅ Final Answer:\n{results[0]}\n")
            print(f"💰 Cost: ${costs[0]:.4f}")
            print(f"🔢 Number of Agents Used: {agent_num_float[0].item():.2f}")
            print(f"📊 Total API Cost: ${Cost.instance().value:.4f}")
            
            # Ask if user wants to continue
            print_separator("-")
            continue_choice = input("\n🔄 Try another query? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes', '']:
                print("\n👋 Thank you for using MasRouter! Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"\n❌ Error: {e}")
            print("Please try again with a different query.")

if __name__ == '__main__':
    main()
