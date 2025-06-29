import os
from openai import OpenAI
import time
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class MultiTurnChatService:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> None:
        if base_url is None:
            base_url=''
   
        if api_key is None:
            api_key='none'


        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.chat_history = []
        self.token_usage = 0

        if system_prompt:
            self.chat_history.append({
                "role": "system",
                "content": system_prompt
            })

    def call(
        self,
        messages: List[Dict[str, Any]],
        model: str = '' 
    ) -> str:
        model = os.environ.get('OPENAI_MODEL', model)

        cnt = 0
        response = None
        while True:
            try:
                cnt += 1
                #print(messages)
                chat_completion = self.client.chat.completions.create(
                    messages=messages,
                    model=model,
                    stream=False,
                    timeout=600,
                    temperature=0.0
                )
                #print(model,'response is',chat_completion)
                response_content = chat_completion.choices[0].message.content
                #print(chat_completion.usage.total_tokens)
                self.token_usage += chat_completion.usage.total_tokens
                self.chat_history.append({
                    "role": "assistant",
                    "content": response_content
                })
                break
            except BaseException as e:
                logger.error(e)
                print('we filed')
                print(len(messages))
                if cnt > 12:
                    logger.error(f'Call to GPT failed after {cnt} attempts')
                    break
                time.sleep(0.5)
        return response_content

    def add_message(self, role: str, content: Any) -> None:
        self.chat_history.append({
            "role": role,
            "content": content
        })

    def import_chat_history(self, history: List[Dict[str, Any]]) -> None:
        self.chat_history = history

    def get_chat_history(self) -> List[Dict[str, Any]]:
        return self.chat_history

    def get_token_usage(self) -> int:
        return self.token_usage

    def chat(self, new_message: Dict[str, Any]) -> str:
        self.add_message(new_message['role'], new_message['content'])
        response = self.call(self.chat_history)
        return response

    def trim_chat_history(self, max_length: int) -> None:
        """Trim the chat history to maintain a certain length without removing the system prompt."""
        print(len(self.chat_history))
        system_prompt = [msg for msg in self.chat_history if msg['role'] == 'system']
        chat_pairs = [msg for msg in self.chat_history if msg['role'] != 'system']

        while len(chat_pairs) > max_length:
            print(len(chat_pairs))
            if len(chat_pairs) >= 2 and chat_pairs[0]['role'] == 'user' and chat_pairs[1]['role'] == 'assistant':
                del chat_pairs[0:2]
            else:
                break

        self.chat_history = system_prompt + chat_pairs

        print(len(self.chat_history))

    def clear_chat_history(self) -> None:
        """Clear the chat history by removing user and assistant pairs."""
        self.chat_history = [msg for msg in self.chat_history if msg['role'] == 'system']

#

chat_service = MultiTurnChatService(
    system_prompt="You are a helpful assistant."
)


# response=chat_service.chat({"role": "user", "content": 'tell me more about it'})
# print("Assistant:", response)
