"""OpenAI model adapter."""
from openai import OpenAI

from ..expander import Expander
from ..schema import CMCHPPacket


class OpenAIAdapter:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.expander = Expander()

    def continue_from_packet(self, packet: CMCHPPacket, user_message: str) -> str:
        system = self.expander.expand(packet, self.model)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content
