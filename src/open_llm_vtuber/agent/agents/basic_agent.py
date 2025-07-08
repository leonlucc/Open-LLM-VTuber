import aiohttp
from typing import AsyncIterator, List, Dict, Any, Literal
from loguru import logger

from .agent_interface import AgentInterface
from ..output_types import SentenceOutput, Actions, DisplayText,VisualizationOutput
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
        """
        {
            "success": true,
            "plan": {{
                "intent": "查询意图描述",
                "query_type": "visualization|answer", 
                "target_table": "主要查询的表名",
                "sql": "生成的SQL语句",
                "display_type": "chart|table|answer",
                "chart_type": "bar|line|pie",
                "x_field": "如果是图表，X轴字段名",
                "y_field": "如果是图表，Y轴字段名",
                "answer_format": "如果是answer类型，如何表述答案",
                "answer_text": "如果是answer类型，返回的答案文本",
                "table_data": [],
                "chart": {}
            }
        }
        """
        #api_result = await self._call_api(user_query)
        api_result = {
            "ai_analysis": "最近的两条销售记录如下",
            "data_source": "database数据，共2条记录",
            "display_type": "table",
            "intent": "查询最近的两条销售记录，用于快速查看最新的销售数据。",
            "query": "查询最近的两条销售记录",
            "query_type": "visualization",
            "sql": "SELECT * FROM sales_data ORDER BY sales_date DESC LIMIT 2;",
            "success": True,
            "table_data": [{
                "category": "食品饮料",
                "created_at": "2025-06-07T21:15:18",
                "id": 25,
                "product_name": "伊利牛奶",
                "quantity": 3,
                "region": "华南",
                "sales_amount": 195.03,
                "sales_date": "2025-05-15",
                "salesperson": "周杰"
            },{
                "category": "美妆个护   ",
                "created_at": "2025-06-07T21:15:18",
                "id": 8,
                "product_name": "欧莱雅洗发水",
                "quantity": 3,
                "region": "华南",
                "sales_amount": 222.03,
                "sales_date": "2025-05-14",
                "salesperson": "周杰"
            }]
        }
        logger.info(f"User query: {user_query}, api_result: {api_result}")
        if not api_result.get("success", False):
            reply = api_result.get("message", "API调用失败")
        elif api_result.get("query_type") == "answer":
            reply = api_result.get("answer_text", "未返回答案")
        elif api_result.get("query_type") == "visualization":
            display_type = api_result.get("display_type", "text")
            if display_type == "chart":
                chart_type = api_result.get("chart_type", "chart")
                reply = f"已生成{chart_type}图。"
                data = api_result.get("chart", {})
            elif display_type == "table":
                reply = "已生成表格。"
                data = api_result.get("table_data", [])
            else:
                reply = "已生成可视化结果。"
                data = {}
        else:
            reply = api_result.get("message", "未知响应类型")

        self._add_message(reply, "assistant")

        # 只返回一句话作为演示
        yield VisualizationOutput(
            display_text=DisplayText(text=reply, name="Agent", avatar=None),
            tts_text=reply,
            display_type=display_type,
            display_data=data,
            actions=Actions()
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

