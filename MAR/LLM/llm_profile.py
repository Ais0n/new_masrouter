# OpenRouter Model Names - Use format: provider/model-name
# Full list available at: https://openrouter.ai/models
llm_profile = [
                {'Name': 'openai/gpt-4o-mini',
                 'Description': 'GPT-4o Mini is a smaller version of the GPT-4o language model, designed for faster inference and reduced memory usage. It retains the same capabilities as the full-size model, but with fewer parameters.\n\
                    The model costs $0.15 per million input tokens and $0.6 per million output tokens\n\
                    In General Q&A Benchmark MMLU, GPT-4o-mini achieves an accuracy of 77.8.\n\
                    In Reasoning Benchmark GPQA, GPT-4o-mini achieves an accuracy of 40.2.\n\
                    In Coding Benchmark HumanEval, GPT-4o-mini achieves an accuracy of 85.7.\n\
                    In Math Benchmark MATH, GPT-4o-mini achieves an accuracy of 66.09.'},
                {'Name': 'anthropic/claude-3.5-haiku',
                 'Description': 'The new Claude 3.5 Haiku combines rapid response times with improved reasoning capabilities, making it ideal for tasks that require both speed and intelligence. Claude 3.5 Haiku improves on its predecessor and matches the performance of Claude 3 Opus.\n\
                    The model costs $1.0 per million input tokens and $5.0 per million output tokens\n\
                    In General Q&A Benchmark MMLU, claude-3-5-haiku achieves an accuracy of 67.9.\n\
                    In Reasoning Benchmark GPQA, claude-3-5-haiku achieves an accuracy of 41.6.\n\
                    In Coding Benchmark HumanEval, claude-3-5-haiku achieves an accuracy of 86.3.\n\
                    In Math Benchmark MATH, claude-3-5-haiku achieves an accuracy of 65.9.'},
                {'Name': 'google/gemini-2.0-flash-001',
                 'Description': 'Gemini 2.0 Flash is the latest generation of Gemini models with 1M context window, purpose-built for speed and efficiency with improved capabilities over 1.5.\n\
                    The model costs $0.10 per million input tokens and $0.40 per million output tokens\n\
                    In General Q&A Benchmark MMLU, gemini-2.0-flash achieves an accuracy of 77.6.\n\
                    In Reasoning Benchmark GPQA, gemini-2.0-flash achieves an accuracy of 60.1.\n\
                    In Coding Benchmark HumanEval, gemini-2.0-flash achieves an accuracy of 85.0.\n\
                    In Math Benchmark MATH, gemini-2.0-flash achieves an accuracy of 90.9.'},
                {'Name': 'meta-llama/llama-3.1-70b-instruct',
                 'Description': 'The Meta Llama-3.1-70b-instruct multilingual large language model (LLM) is a pretrained and instruction tuned generative model in 70B (text in/text out).\n\
                    The model costs $0.2 per million input tokens and $0.2 per million output tokens\n\
                    In General Q&A Benchmark MMLU, Llama 3.1 achieves an accuracy of 79.1.\n\
                    In Reasoning Benchmark GPQA, Llama 3.1 achieves an accuracy of 46.7.\n\
                    In Coding Benchmark HumanEval, Llama 3.1 achieves an accuracy of 80.7.\n\
                    In Math Benchmark MATH, Llama 3.1 achieves an accuracy of 60.3.'},
                {'Name': 'deepseek/deepseek-chat',
                 'Description': 'DeepSeek-V3 is a powerful open-source Mixture-of-Experts (MoE) language model developed by Chinese AI company DeepSeek, featuring 671 billion total parameters with 37 billion activated per token, achieving performance comparable to leading closed-source models like GPT-4.\n\
                    The model costs $0.27 per million input tokens and $1.1 per million output tokens\n\
                    In General Q&A Benchmark MMLU, deepseek-chat achieves an accuracy of 88.5.\n\
                    In Reasoning Benchmark GPQA, deepseek-chat achieves an accuracy of 59.1.\n\
                    In Coding Benchmark HumanEval, deepseek-chat achieves an accuracy of 88.4.\n\
                    In Math Benchmark MATH, deepseek-chat achieves an accuracy of 85.1'},
                ]