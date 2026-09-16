"""Claude model adapter — send expanded handoff context and get response."""
import anthropic

from ..expander import Expander
from ..schema import CMCHPPacket


class ClaudeAdapter:
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.expander = Expander()

    def continue_from_packet(self, packet: CMCHPPacket, user_message: str) -> str:
        system = self.expander.expand(packet, self.model)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text
