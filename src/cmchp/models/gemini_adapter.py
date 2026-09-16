"""Gemini model adapter."""
import google.generativeai as genai

from ..expander import Expander
from ..schema import CMCHPPacket


class GeminiAdapter:
    def __init__(self, api_key: str | None = None, model: str = "gemini-2.0-flash"):
        if api_key:
            genai.configure(api_key=api_key)
        self.model_name = model
        self.expander = Expander()

    def continue_from_packet(self, packet: CMCHPPacket, user_message: str) -> str:
        system = self.expander.expand(packet, self.model_name)
        model = genai.GenerativeModel(
            self.model_name,
            system_instruction=system,
        )
        response = model.generate_content(user_message)
        return response.text
