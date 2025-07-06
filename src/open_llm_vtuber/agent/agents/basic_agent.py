import aiohttp
from typing import AsyncIterator, List, Dict, Any, Literal
from loguru import logger

from .agent_interface import AgentInterface
from ..output_types import SentenceOutput, DisplayText
from ..input_types import BatchInput, TextSource

class BasicAgent(AgentInterface):
    """
    Basic agent that chats by calling a local HTTP API.
    """

    def __init__(
        self,
        live2d_model=None,
        tts_preprocessor_config=None,
        faster_first_response: bool = True,
        segment_method: str = "pysbd",
        interrupt_method: Literal["system", "user"] = "user",
    ):
        super().__init__()
        self._live2d_model = live2d_model
        self._tts_preprocessor_config = tts_preprocessor_config
        self._faster_first_response = faster_first_response
        self._segment_method = segment_method
        self.interrupt_method = interrupt_method
        self._memory = []
        logger.info("BasicAgent initialized.")

    def _add_message(self, message: str, role: str):
        self._memory.append({"role": role, "content": message})

    async def _call_api(self, query: str) -> Dict[str, Any]:
        url = "http://localhost:5000/api/query"
        payload = {"query": query}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.error(f"API call failed: {resp.status}")
                    return {"success": False, "message": f"HTTP {resp.status}"}

    def _extract_user_query(self, input_data: BatchInput) -> str:
        # 只取第一个输入文本
        for text_data in input_data.texts:
            if text_data.source == TextSource.INPUT:
                return text_data.content
        return ""

    async def chat(self, input_data: BatchInput) -> AsyncIterator[SentenceOutput]:
        user_query = self._extract_user_query(input_data)
        self._add_message(user_query, "user")

        api_result = await self._call_api(user_query)
        if not api_result.get("success", False):
            reply = api_result.get("message", "API调用失败")
        elif api_result.get("query_type") == "answer":
            reply = api_result.get("answer_text", "未返回答案")
        elif api_result.get("query_type") == "visualization":
            display_type = api_result.get("display_type")
            if display_type == "chart":
                chart_type = api_result.get("chart_type", "chart")
                reply = f"已生成{chart_type}图。"
            elif display_type == "table":
                reply = "已生成表格。"
            else:
                reply = "已生成可视化结果。"
        else:
            reply = api_result.get("message", "未知响应类型")

        self._add_message(reply, "assistant")

        # 只返回一句话作为演示
        yield SentenceOutput(
            text=reply,
            display=DisplayText(name="Agent", avatar=None)
        )

    
    def handle_interrupt(self, heard_response: str) -> None:
        """
        Handle user interruption (not implemented for BasicAgent).
        """
        pass

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        """
        Set memory from chat history (not implemented for BasicAgent).
        """
        pass

