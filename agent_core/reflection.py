"""
反思模块
任务执行失败时的自我反思与策略调整
"""
from typing import List, Dict, Any, Optional
import json
import logging

from langchain_community.llms import Tongyi

logger = logging.getLogger(__name__)

#反思模块 分析失败原因并生成替代策略
class Reflection:
    """反思器 - 分析失败原因并生成替代策略"""
    
    def __init__(self, llm: Optional[Tongyi] = None):
        self.llm = llm#大模型初始化
        self.max_retries = 3#规定最大的反思次数是3 防止无限次反思循环

#失败分析调度入口：接收失败的任务和错误信息，返回调整后的任务参数。
    def analyze_failure(self, task: Dict[str, Any], error: str) -> Dict[str, Any]:
        """
        分析单个任务失败原因，返回调整后的任务参数
        
        Args:
            task: 失败的任务字典
            error: 错误信息
        
        Returns:
            调整后的任务（可能是原参数或修改后的参数）
        """
        if self.llm:#先让 LLM 分析
            return self._llm_based_analyze(task, error)
        else:# 没 LLM 就规则修正
            return self._rule_based_analyze(task, error)

#大模型分析任务失败的原因    
    def _llm_based_analyze(self, task: Dict[str, Any], error: str) -> Dict[str, Any]:#接收的是执行失败的某个任务步骤，类型是字典；和错误
        """基于LLM分析"""
        prompt = f"""任务执行失败，请分析错误原因并提供修正后的任务参数。
原始任务：{json.dumps(task, ensure_ascii=False)}
错误信息：{error}

请返回一个JSON对象，包含 tool 和修正后的 params。例如：
{{"tool": "data_analyzer", "params": {{"action": "read_data", "file_path": "data/sales_data.xlsx"}}}}
确保只返回有效的JSON。"""
        
        try:
            response = self.llm.invoke(prompt)#大模型根据提示词的输出
            # 提取JSON对象
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                adjusted = json.loads(response[start:end])
                return adjusted#返回经过 LLM 分析后，修正过的任务字典。
        except Exception as e:
            logger.error(f"LLM反思分析失败: {e}")#发生异常了 记录日志
        
        # JSON 解析失败，降级到 _rule_based_analyze()方法
        return self._rule_based_analyze(task, error)

#降级执行的规则修正方法 没有大模型的帮助  它不重新分析复杂原因，只根据错误文本和工具名，对少数常见参数做固定修改。 
    def _rule_based_analyze(self, task: Dict[str, Any], error: str) -> Dict[str, Any]:
        """基于规则修正"""
        adjusted = task.copy()
        error_lower = error.lower()
        
        if task["tool"] == "data_analyzer":
            if "file" in error_lower and ("not found" in error_lower or "不存在" in error):
                adjusted["params"]["file_path"] = "data/sales_data.xlsx"
            elif "action" in error_lower:
                adjusted["params"]["action"] = "read_data"  # 确保有效action
        
        elif task["tool"] == "api_tool":
            if "system" in error_lower:
                adjusted["params"]["system"] = "OA"  # 默认使用OA
#复制原任务并按固定规则修正参数，例如 data_analyzer 文件不存在时改用 data/sales_data.xlsx、action 异常时改为 read_data，api_tool 的 system 异常时改为 OA，最后返回调整后的任务       
        return adjusted

#判断是否重试方法 一般放在分析失败之后
    def should_retry(self, error: str, retry_count: int) -> bool:#判断该不该重试 如果有这些敏感词的话就重试
        """判断是否应该重试"""
        if retry_count >= self.max_retries:#已重试 3 次：不再重试 小于3次才重试
            return False
        #定义可重试错误关键词
        retryable_errors = ["timeout", "connection", "network", "暂时", "超时", "重试"]
        error_lower = error.lower()
        return any(err in error_lower for err in retryable_errors)

#放在整体计划失败的时候  用来生成替代执行的计划
    def generate_alternative_plan(self, original_tasks: List[Dict], error_context: str) -> List[Dict]:#换一条路走 整个计划都失败时，让 LLM 重新生成一套任务步骤。 LLM 不行就降级成一个极简方案：把错误描述丢给知识问答工具，至少让用户得到点反馈。
        """
        当整体任务失败时，生成替代执行计划
        
        Args:
            original_tasks: 原始任务列表
            error_context: 错误上下文
        
        Returns:
            替代任务列表
        """
        if self.llm:
            prompt = f"""原始任务计划遇到问题，请生成一个替代方案。
原始计划：{json.dumps(original_tasks, ensure_ascii=False)}
错误描述：{error_context}

请返回一个新的任务步骤列表，JSON数组格式。"""
            try:
                response = self.llm.invoke(prompt)
                start = response.find('[')
                end = response.rfind(']') + 1
                if start != -1 and end > start:
                    return json.loads(response[start:end])
            except Exception as e:
                logger.error(f"LLM替代计划生成失败: {e}")
        
        # 降级：返回简化计划
        return [
            {"tool": "knowledge_qa", "params": {"question": error_context}}
        ]
        
#放在整个任务  结束了之后 总结成功、失败和改进的地方
    def reflect_on_execution(self, tasks: List[Dict], results: List[Any]) -> str:#复盘总结经验 把“任务列表 + 执行结果”给 LLM，让它总结成功之处和改进点。 LLM 不可用时，做个简单统计：“成功 N 项，失败 M 项。”
        """对整个执行过程进行反思，输出总结经验"""
        if self.llm:
            prompt = f"""请对以下任务执行过程进行反思总结，指出成功之处和可改进点。
任务：{json.dumps(tasks, ensure_ascii=False)}
结果：{results}

总结："""
            try:
                return self.llm.invoke(prompt).strip()
            except Exception as e:
                logger.error(f"LLM反思总结失败: {e}")
        
        # 简单统计
        success = sum(1 for r in results if not str(r).startswith("Error") and "失败" not in str(r))
        return f"任务执行完成：成功 {success} 项，失败 {len(results)-success} 项。"