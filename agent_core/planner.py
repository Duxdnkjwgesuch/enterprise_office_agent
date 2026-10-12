"""
任务规划器
根据用户指令自动拆解为多个子任务步骤
支持 LLM 智能规划和规则降级策略
"""
from typing import List, Dict, Any, Optional
import json
import logging
from langchain_community.llms import Tongyi
from config.settings import settings

logger = logging.getLogger(__name__)


class TaskPlanner:
    """任务规划器 - 将复杂任务拆解为可执行的步骤"""

    def __init__(self, llm: Optional[Tongyi] = None):#初始化 需要大模型
        self.llm = llm
        
#plan方法 形参：用户的输入，当前 Agent 已注册、可以使用的工具名称列表，用来告诉 Planner 只能从这些工具里制定计划 
    def plan(self, user_input: str, available_tools: List[str]) -> List[Dict[str, Any]]:
        """
        规划任务步骤：优先使用 LLM，失败时回退到规则
        """
        # 1. 尝试 LLM 规划 优先让大模型（LLM）拆解：把“有哪些工具可用 + 用户说了什么”一起发给通义千问，让它输出一个 JSON 格式的步骤清单。
        if self.llm:
            llm_plan = self._llm_based_plan(user_input, available_tools)
            if llm_plan:
                return llm_plan#返回大模型规划的计划步骤清单
        # 2. 规则降级
        return self._rule_based_plan(user_input)#如果出现 没有配置 LLM、LLM 规划失败或超时、LLM 返回的内容不是合法 JSON、LLM 返回的计划为空这些情况会执行这一行代码 就是_rule_based_plan方法。

#LLM 拆解计划是怎么做的方法
    def _llm_based_plan(self, user_input: str, available_tools: List[str]) -> Optional[List[Dict[str, Any]]]:
        """基于LLM进行规划，失败返回 None"""
        tools_str = ", ".join(available_tools)#可用的工具列表，提示词（根据用户的输入拆解请求，每个步骤使用工具解决，输出json格式的步骤列表）
        prompt = f"""你是一个任务规划专家。用户请求需要使用以下工具来完成任务：
可用工具：{tools_str}

用户请求：{user_input}

请分析请求，将任务拆解为具体的步骤。每个步骤必须使用上述工具之一，输出JSON格式的步骤列表。
示例格式：
[
    {{"tool": "data_analyzer", "params": {{"action": "read_data", "file_path": "data/sales_data.xlsx"}}}},
    {{"tool": "data_analyzer", "params": {{"action": "analyze"}}}},
    {{"tool": "data_analyzer", "params": {{"action": "generate_report", "auto_generate_doc": true}}}},
]

确保输出有效的JSON数组，不要包含任何额外文字。"""

        try:
            response = self.llm.invoke(prompt)#大模型根据提示词输出的结果
            start = response.find('[')#我们需要的结果开头
            end = response.rfind(']') + 1#我们需要的结果结尾
            if start != -1 and end > start:
                json_str = response[start:end]
                tasks = json.loads(json_str)
                logger.info(f"LLM规划生成 {len(tasks)} 个步骤")
                return tasks#tasks是返回的最终json
        except Exception as e:
            logger.error(f"LLM规划失败: {e}")#出现异常了返回失败
        return None#连try都没进去就返回None

#从用户的自然语言中提取 Excel/CSV 文件路径，供规则 Planner 使用。
    def _extract_file_path(self, user_input: str) -> str:
        """从用户输入中提取文件路径（供规则使用）"""
        import re
        patterns = [
            r'文件路径\s*[：:]\s*([^\s]+)',
            r'path\s*[：:]\s*([^\s]+)',
            r'文件\s*[：:]\s*([^\s]+)',
            r'(?:分析|查看|处理|使用|打开|读取|路径)?[：:\s]*(\S+\.(xlsx|xls|csv))',
            r'([^\s]+\.(xlsx|xls|csv))',
        ]
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                return match.group(1)
        return "data/sales_data.xlsx"#如果所有正则都没有匹配到，就返回默认路径

#兜底的方法 当 LLM 不可用、规划失败或输出不是合法 JSON 时，用关键词把用户请求转换成固定的工具执行步骤，用来在 LLM 规划失败时保证常见任务仍能执行。
    def _rule_based_plan(self, user_input: str) -> List[Dict[str, Any]]:
        """基于规则的规划（兜底）"""
        user_input_lower = user_input.lower()#把输入转成小写，用于匹配 excel、oa、crm 等英文词。
        file_path = self._extract_file_path(user_input)#提取用户输入中的 Excel/CSV 路径；没有提取到就使用 data/sales_data.xlsx。
#包含“分析”，同时包含 Excel、数据或销售：读取、清洗、分析、画图、生成报告
        if "分析" in user_input and ("excel" in user_input_lower or "数据" in user_input_lower or "销售" in user_input_lower):
            return [
                {"tool": "data_analyzer", "params": {"action": "read_data", "file_path": file_path}},
                {"tool": "data_analyzer", "params": {"action": "clean_data"}},
                {"tool": "data_analyzer", "params": {"action": "analyze"}},
                {"tool": "data_analyzer", "params": {"action": "visualize"}},
                {"tool": "data_analyzer", "params": {"action": "generate_report", "auto_generate_doc": True}}
            ]
#包含“报告”和“生成”：读取数据、分析、生成报告文档
        elif "报告" in user_input and "生成" in user_input:
            return [
                {"tool": "data_analyzer", "params": {"action": "read_data", "file_path": file_path}},
                {"tool": "data_analyzer", "params": {"action": "analyze"}},
                {"tool": "document_generator", "params": {"doc_type": "report", "auto_generate": True, "title": "数据分析报告"}}
            ]
#包含“会议纪要”或“会议记录”：调用文档工具生成会议纪要
        elif "会议纪要" in user_input or "会议记录" in user_input:
            return [
                {"tool": "document_generator", "params": {
                    "doc_type": "meeting_notes", "title": "会议纪要",
                    "auto_generate": True, "context": user_input
                }}
            ]
#包含“工作总结”“周报”或“月报”：调用文档工具生成工作总结
        elif "工作总结" in user_input or "周报" in user_input or "月报" in user_input:
            return [
                {"tool": "document_generator", "params": {
                    "doc_type": "work_summary", "title": "工作总结",
                    "auto_generate": True, "context": user_input
                }}
            ]
#包含“审批”或“申请”：调用 OA 发起审批
        elif "审批" in user_input or "申请" in user_input:
            system = "OA"
            approval_type = "leave"
            if "报销" in user_input: approval_type = "expense"
            elif "合同" in user_input: approval_type = "contract"
            elif "采购" in user_input: approval_type = "purchase"
            elif "出差" in user_input: approval_type = "travel"
            elif "加班" in user_input: approval_type = "overtime"
            return [{"tool": "api_tool", "params": {
                "action": "process_approval", "system": system,
                "approval_type": approval_type, "use_mock": True
            }}]
#包含 OA 或 CRM，并且包含查看或状态：查询 OA/CRM 状态
        elif "oa" in user_input_lower or "crm" in user_input_lower:
            system = "OA" if "oa" in user_input_lower else "CRM"
            if "状态" in user_input or "查看" in user_input:
                return [{"tool": "api_tool", "params": {"action": "get_system_status", "system": system}}]
            elif "审批" in user_input or "申请" in user_input:
                return [{"tool": "api_tool", "params": {
                    "action": "process_approval", "system": system,
                    "approval_type": "leave", "use_mock": True
                }}]
#包含知识、制度、政策或产品：调用知识问答工具
        elif "知识" in user_input or "制度" in user_input or "政策" in user_input or "产品" in user_input:
            return [{"tool": "knowledge_qa", "params": {"question": user_input}}]
        else:
            # 以上都没有命中默认执行知识问答
            return [{"tool": "knowledge_qa", "params": {"question": user_input}}]
