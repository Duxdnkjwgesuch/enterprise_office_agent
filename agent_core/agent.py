"""
核心Agent封装
基于LangChain ReAct Agent，集成记忆、工具调用、反思功能
支持复杂任务的自动规划与多步骤执行
"""
import os
import uuid
from typing import Optional, List, Dict, Any
import logging

from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import Tongyi
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

from config.settings import settings
from .memory_manager import MemoryManager
from .tool_registry import ToolRegistry
from .reflection import Reflection
from .planner import TaskPlanner

logger = logging.getLogger(__name__)

#父类的滑动窗口记忆只保留近k轮的记忆 超出的丢掉 服务重启，所有对话全部消失，无法恢复。
#我们继承重写之后的增强 1.持久化落库：哪怕内存里旧对话被滑动窗口删掉了，数据库里完整保存着全部历史消息；服务重启，根据 session_id 可以重新读取历史，恢复会话。
#2.对接自定义 MemoryManager：每一轮对话保存的时候，自动调用记忆管理器写入消息，方便统一管理会话、查询历史。
class CustomConversationMemory(ConversationBufferWindowMemory):#创建一个 名为 自定义对话记忆的类 继承自Lnagchain的滑动窗口记忆
    """扩展记忆类，在保存上下文时同步到 MemoryManager，并避免多 key 警告"""
    memory_manager: MemoryManager = None #类属性 记忆管理默认为空
    session_id: str = ""                   #类属性 session_id默认为空字符串  区分不同用户的对话会话，每条消息都会绑定这个会话 id

    class Config:#基于 Pydantic 做模型校验。
        arbitrary_types_allowed = True#允许这个类里面存放自定义对象（MemoryManager），不让 Pydantic 抛类型校验异常。
#不加这一行，Pydantic 会报错，因为它不认识 MemoryManager 这个自定义类。

#每一轮对话结束 自动触发 将把这一轮「用户提问 + 模型回答」保存进记忆。
    def save_context(self, inputs: dict, outputs: dict) -> None:#重写父类中保存上下文的方法 两个形参 存放用户输入的内容 和 存放大模型输出的内容
        output_text = outputs.get("output", "")#取出 output 里模型返回的内容
        clean_outputs = {"output": output_text}#只保留output字段，封装成新字典传给父类
        super().save_context(inputs, clean_outputs)#依旧保留滑动窗口的优势 只保留近k轮 不会改变传给大模型的上下文长度 控制token开销

        if self.memory_manager:#如果记忆存在
            user_input = inputs.get("input", "")#得到user的输入内容
            if user_input:#如果用户有输入
                self.memory_manager.add_message(self.session_id, "user", user_input)#将这个输入存入对应的session_id中
            if output_text:#如果有大模型输出
                self.memory_manager.add_message(self.session_id, "assistant", output_text)#同理


class OfficeAgent:#封装了完整的 ReAct Agent 执行流程这个模板告诉LLM：你应该如何思考、如何行动、如何观察、何时结束。"
    REACT_TEMPLATE = """你是一个企业智能办公助手，能够使用各种工具帮助用户完成办公任务。

你可以使用的工具有：
{tools}

工具名称列表：{tool_names}

请严格按照以下格式回答：

Question: 用户的问题
Thought: 思考需要做什么
Action: 要使用的工具名称（必须是 [{tool_names}] 中的一个）
Action Input: 工具的输入参数（JSON格式）
Observation: 工具返回的结果
... (这个 Thought/Action/Action Input/Observation 可以重复多次)
Thought: 我现在知道最终答案了
Final Answer: 对用户的最终回复

开始！

历史对话：
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""#类属性，所有的对象都可以调用

    def __init__(self, llm: Optional[Tongyi] = None, enable_planner: bool = True):#两个默认参数 大模型和任务规划器 任务规划器是默认打开的
        if llm:#每次创建智能体对象的时候 都会先检查是否存在大模型 如果不存在就去初始化它
            self.llm = llm
        else:
            self._init_llm()

        self.memory_manager = MemoryManager()#创建记忆管理器对象
        self.tool_registry = ToolRegistry()#创建工具调用器对象
        self.reflection = Reflection(self.llm)#创建反思模块对象，参数是大模型传给 反思模块的init 也就是专门用来做反思的大模型
        self.agent_executor: Optional[AgentExecutor] = None#定义agent_executor变量 这个变量可以是AgentExecutor类型 也可以是 None 本意是可以复用这个选择器，但是通篇都没用过
        self.enable_planner = enable_planner and (self.llm is not None)#任务规划器 两边都为true 才行 又得有大模型又得有任务规划器才真正开启
        if self.enable_planner:#如果任务规划器被开启 就执行TaskPlanner对象
            self.task_planner = TaskPlanner(self.llm)
        logger.info(f"OfficeAgent初始化完成，Planner模式: {'开启' if self.enable_planner else '关闭'}")#记录日志

    def _init_llm(self):#初始化大模型 得到apikey 模型名字等
        if settings.DASHSCOPE_API_KEY:
            os.environ["DASHSCOPE_API_KEY"] = settings.DASHSCOPE_API_KEY
            self.llm = Tongyi(
                model_name=settings.LLM_MODEL_NAME,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS
            )
            logger.info(f"LLM初始化成功，模型: {settings.LLM_MODEL_NAME}")
        else:
            logger.warning("未配置API Key，Agent将无法正常工作")
            self.llm = None

    def register_tool(self, tool):#注册单个工具，将参数交给tool_registry.register(tool)，也就是ToolRegistry类中的register方法 具体逻辑在这个里面实现
        self.tool_registry.register(tool)

    def register_tools(self, tools: List):#批量注册，一次传入工具列表，注册多个
        self.tool_registry.register_many(tools)

#内部私有方法 只能在这个类中使用。
#负责给单步 ReAct 创建一个配置好工具、Prompt 和记忆的执行器。
    def _create_agent_executor(self, session_id: str) -> AgentExecutor:#这个方法的功能是：根据当前会话 session_id，新建一份绑定该会话记忆的 ReAct Agent 执行器，用来跑单步 ReAct 任务。
        tools = self.tool_registry.list_tools()#获取所有的工具列表，包含名称等。工具
        if not tools:
            raise ValueError("没有注册任何工具，请先注册工具")#检查工具列表中是否有工具存在

        prompt = PromptTemplate.from_template(self.REACT_TEMPLATE)#提示词模版是REACT_TEMPLATE。 提示词
        memory = CustomConversationMemory(#创建继承滑动窗口自定义对话类的对象叫 memory 传入6个参数
            memory_key="chat_history",#指定记忆在提示词模版中的变量是一致的，防止提示词模版找不见。
            return_messages=False,#控制历史记录以什么类型返回，False：返回拼接好的字符串。True：返回 HumanMessage、AIMessage 等消息对象列表。当前 ReAct 模板是普通字符串 Prompt
            input_key="input",#告诉记忆对象：用户输入应该从input字段开始读取。
            k=settings.SHORT_TERM_MEMORY_MAX_LENGTH,#控制滑动窗口保留多少历史 我的配置是10
            memory_manager=self.memory_manager,#定义的记忆管理对象 。Redis 短期记忆，MySQL 持久化记录，进程内降级缓存。
            session_id=session_id#绑定session——id。指定当前记忆属于哪个会话，因此不同用户的对话不会写进同一个记忆空间。
        )

        history = self.memory_manager.get_messages(session_id)#历史记录 先从项目的 MemoryManager 中读取当前会话的历史记录。session_id 决定读取哪个用户或哪次会话。
        for msg in history:#遍历历史记录 并判断那个属于user那个属于ai
            if msg["role"] == "user":
                memory.chat_memory.add_user_message(msg["content"])
            else:
                memory.chat_memory.add_ai_message(msg["content"])

        agent = create_react_agent(llm=self.llm, tools=tools, prompt=prompt)#agent 形参是大模型 工具 提示词
        executor = AgentExecutor(#执行器 包含agent 工具 记忆 打印 Agent 内部详细日志 捕获 LLM 输出解析异常
            agent=agent, tools=tools, memory=memory,
            verbose=settings.DEBUG, handle_parsing_errors=True,
            max_iterations=8, return_intermediate_steps=True#防止死循环Agent 最多能循环执行多少轮思考 + 工具调用；保存每一步中间执行记录
        )
        return executor#返回这个执行器

    def _is_complex_task(self, user_input: str) -> bool:#只要包含以下这些词 就定义为复杂任务
        complex_keywords = ["分析", "报告", "生成", "图表", "可视化", "工作总结", "会议纪要", "审批"]
        return any(kw in user_input for kw in complex_keywords)#用户输入包含以下的词汇
#多步复杂方法
    def _execute_planned_task(self, user_input: str, session_id: str,
                                user_id: Optional[str] = None) -> Dict[str, Any]:#计划任务执行器
        """多步骤任务执行（使用规划器）"""
        available_tools = self.tool_registry.get_tool_names()#获取所用工具的名称
        task_steps = self.task_planner.plan(user_input, available_tools)#TaskPlanner类中的plan方法 根据用户的输入和工具列表来判断是否能完成任务
        if not task_steps:#如果不能够完成就降级成单步任务 _execute_single_step
            return self._execute_single_step(user_input, session_id, user_id)

        logger.info(f"任务规划成功，共 {len(task_steps)} 个步骤")#记录日志
        all_steps = []#初始化结果容器
        final_output = ""#初始化最终的输出

        for i, step in enumerate(task_steps):#迭代plan方法
            tool_name = step.get("tool")
            params = step.get("params", {})#获取工具名称和执行路径
            logger.info(f"执行步骤 {i+1}/{len(task_steps)}: {tool_name}")#记录到日志中

            tool = self.tool_registry.get_tool(tool_name)#根据工具名称，从 ToolRegistry 中获取对应工具实例。
            if not tool:#如果planner输出了一个不存在的工具，系统不会崩溃，而是输出错误未找到工具
                error_msg = f"错误：未找到工具 {tool_name}"
                all_steps.append({"step": i+1, "tool": tool_name, "error": error_msg})#将刚才一步的信息添加到结构容器中
                final_output += f"\n[步骤{i+1} 失败] {error_msg}"#添加到最终输出中
                continue#使用 continue 跳过当前步骤，继续执行后面的步骤

            try:
                # 修复：使用标准调用 tool.invoke(params)
                observation = tool.invoke(params)#真正执行工具
                all_steps.append({#执行之后在结构容器中添加对应的信息：工具名称，路径，完成任务的内容
                    "tool": tool_name,
                    "tool_input": params,
                    "observation": observation
                })
                final_output += f"\n[步骤{i+1} 完成] {tool_name}: {observation[:200]}"#将成功的结果反馈给最终输出中，拼接用户可读的结果
                self.memory_manager.mysql.log_tool_call(#在记忆的mysql持久化记忆中，记录session- ID，user- ID，工具名称，就是 Planner 为当前步骤生成的工具参数字典，工具修改的内容，只有成功执行的工具才会写入 MySQL
                    session_id=session_id,
                    user_id=user_id or "anonymous",
                    tool_name=tool_name,
                    params=params,
                    result=str(observation)[:1000]
                )
            except Exception as e:#抛出异常 告知用户出错的地方 并写入日志
                error_msg = f"执行出错: {str(e)}"
                logger.error(f"步骤 {i+1} 执行失败: {e}")
                all_steps.append({"step": i+1, "tool": tool_name, "error": error_msg})
                final_output += f"\n[步骤{i+1} 出错] {error_msg}"

        return {#返回
            "output": final_output.strip() or "任务执行完成",#给用户看的最终文字
            "intermediate_steps": all_steps,#每一步的结构化执行结果。
            "session_id": session_id,#本次的会话编号
            "planned": True,#表示走了 Planner 路径。
            "steps_count": len(task_steps)#计划中的步骤数量。
        }
#单步任务方法
    def _execute_single_step(self, user_input: str, session_id: str,
                                user_id: Optional[str] = None) -> Dict[str, Any]:
        """单步任务（ReAct Agent）"""
        try:
            executor = self._create_agent_executor(session_id)#调用_create_agent_executo这个方法 接受对应的session-id
            user_context = ""#初始化用户的内容
            if user_id:#如果用户ID之前就存在 我们从记忆中读取他之前的用户偏好
                prefs = self.memory_manager.get_all_preferences(user_id)
                if prefs:#如果这个用户有用户偏好得话
                    user_context = f"\n用户偏好信息：{prefs}"#将他偏好的内容赋值给user——context

            result = executor.invoke({"input": user_input + user_context})#返回的结果包含用户的输入和他的偏好
            steps = result.get("intermediate_steps", [])#从结果中取出 ReAct 执行过程中调用过的工具步骤；如果没有这个字段，就返回空列表 []。
            for step in steps:#遍历这个工具步骤
                if len(step) >= 2:#每个步骤是否至少包含两个元素：step[0]：AgentAction，包含工具名和参数；step[1]：工具返回的 observation
                    action, observation = step[0], step[1]
                    self.memory_manager.mysql.log_tool_call(#将这些写入mysql中
                        session_id=session_id,#session-id
                        user_id=user_id or "anonymous",
                        tool_name=action.tool,#记录哪个工具被调用、参数是什么、返回结果是什么，结果最多保存 1000 个字符。
                        params=action.tool_input,
                        result=str(observation)[:1000]
                    )
            return {#返回 反馈
                "output": result.get("output", "执行完成"),
                "intermediate_steps": steps,
                "session_id": session_id,
                "planned": False
            }
        except Exception as e:
            logger.error(f"单步执行失败: {e}", exc_info=True)#将失败的原因写入日志
            return {#反馈失败的信息
                "output": f"执行失败: {str(e)}",
                "intermediate_steps": [],
                "session_id": session_id,
                "error": str(e)
            }
#ReAct 模式使用 AgentExecutor，Planner 模式不使用 AgentExecutor。但 Planner 失败并降级时，会重新进入 ReAct，因此仍然可能触发 _create_agent_executor()。
#执行顺序，此方法要接收到用户的输入，session-id，用户id，执行模式的开关：None：没有指定，系统自动判断，True：强制执行 Planner 多步骤规划模式。False：强制执行 ReAct 单步模式。
    def run(self, user_input: str, session_id: Optional[str] = None,
            user_id: Optional[str] = None, use_planner: bool = None) -> Dict[str, Any]:
        if session_id is None:#如果会话id是空的 就随机分配一个
            session_id = str(uuid.uuid4())
        if not self.llm:#如果大模型不存在 就反馈给用户
            return {
                "output": "LLM未初始化，请配置API Key后重试。",
                "intermediate_steps": [],
                "session_id": session_id
            }
        if use_planner is None:#自动进行逻辑判断
            use_planner = self.enable_planner and self._is_complex_task(user_input)#Planner 已启用 并且 用户的输入包含复杂词汇 才会自动使用Planner
        if use_planner:#根据结果分流
            logger.info("使用任务规划器处理请求")#记录日志
            return self._execute_planned_task(user_input, session_id, user_id)#返回结果
        else:
            logger.info("使用ReAct Agent处理请求")#记录日志
            return self._execute_single_step(user_input, session_id, user_id)#返回结果

    def clear_session(self, session_id: str):#清空会话方法
        self.memory_manager.clear_session(session_id)