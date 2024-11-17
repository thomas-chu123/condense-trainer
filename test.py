import requests
from transformers import AutoTokenizer, pipeline

class ConvoGenerator:
    def __init__(self, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", api_key=None):
        self.model_id = model_id
        self.together_model_id = f"{model_id}-Turbo"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.pipeline = pipeline("text-generation", model=model_id, tokenizer=self.tokenizer)
    
    def _get_assistant_messages(self, messages):
        a_messages = messages[2:]  # Skip system and initial user message
        for i in range(len(a_messages)):
            if a_messages[i]["role"] == "assistant":
                a_messages[i]["role"] = "user"
            else:
                a_messages[i]["role"] = "assistant"
        return a_messages
    
    def generate_conversation(
            self, 
            context_seed: str = "Manchester United is a football club. They play in the Premier League.",
            initial_system_message="You are the most intelligent AI in the world. Your task is validate the user's knowledge and always raise new questions or topics. Let's focus on {context_seed}.", 
            initial_user_message="I'm a helpful assistant. Please ask me anything.",
            max_turns: int = 3,
            max_tokens: int = 4096,
    ):
        initial_system_message = initial_system_message.format(context_seed=context_seed)
        # Initialize conversation
        messages = [
            {"role": "system", "content": initial_system_message},
            {"role": "user", "content": initial_user_message},
        ]
        
        total_tokens = 0
        # Get first response
        text = self.generate(messages)
        messages.append({"role": "assistant", "content": text})
        total_tokens += len(self.tokenizer.encode(text))
        
        # Generate multiple conversation turns
        assistant_messages = self._get_assistant_messages(messages)
        for _ in range(max_turns):
            if total_tokens > max_tokens:
                break
                
            text = self.generate(assistant_messages)
            assistant_messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": text})
            total_tokens += len(self.tokenizer.encode(text))
            
            if total_tokens > max_tokens:
                break
                
            text = self.generate(assistant_messages)
            assistant_messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": text})
            total_tokens += len(self.tokenizer.encode(text))

        return {
            "original_conversation": messages,
            "assistant_conversation": assistant_messages,
            "total_tokens": total_tokens
        }
    
    def generate(self, messages, max_new_tokens=1024):
        output = self.pipeline(messages, max_new_tokens=max_new_tokens)
        return output[0]["generated_text"][-1]["content"]

if __name__ == "__main__":
    
    convo_generator = ConvoGenerator(
        api_key="",
        model_id="meta-llama/Llama-3.2-3B-Instruct"
    )
    context_seed = """
By Grant Mortenson Recent events on campus — not the least of which is the article penned in last month’s issue of the Clarion, “Safe Spaces, Brave Places” — have underscored the existence of multiple worldviews here at Bethel University. A worldview is a lens through which each person observes the events occurring around them and by which they explain the significance of everything they encounter. While we should not fear interacting with people who hold a different worldview than our own, there are some that cannot and will not coexist with each other. When worldviews conflict, one will trump the other (cf. Matt. 6:24). In the larger Western culture—and increasingly at Bethel—there is a conflict between two particular worldviews: Biblical Christianity and the Sexual Revolution. The sexual revolutionaries have been calling for ‘tolerance’ and ‘inclusivity’ at Bethel for a few years now. This is not exclusively the worldview of the LGBT community, though they are by far the most vocal proponents of it. Rather, the sexual revolution began with the concept of no-fault divorce and every change that followed can be traced to the misconception and devaluation of marriage. The primary tenets of the sexual revolution are these: “you may do with your body whatever you please,” and “you are free to love whomever you please, so long as all parties consent.” Both fly squarely in the face of Biblical Christianity and undermine some of the most basic principles in understanding our relationship to Christ.
"""
    convo = convo_generator.generate_conversation(context_seed=context_seed)

    print(convo)

    import json

    with open("convo.json", "w") as f:
        json.dump(convo, f, indent=4)