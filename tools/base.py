"""
工具基类定义
所有自定义工具都继承自 LangChain 的 BaseTool
"""
from langchain_core.tools import BaseTool as LangChainBaseTool
from abc import abstractmethod
from typing import Any, Optional, Type
from pydantic import BaseModel

#BaseTool 是这个项目所有业务工具的统一基类。它继承 LangChain 的 BaseTool，保证四个工具都能以统一方式注册和调用。
class BaseTool(LangChainBaseTool):#继承LangChainBaseTool
    """
    工具基类，扩展了 LangChain BaseTool
    
    子类需要实现：
    - name: 工具名称
    - description: 工具描述
    - _execute: 具体的业务逻辑（接收参数）
    - _run: 同步执行方法（已自动处理参数）
    - _arun: 异步执行方法（可选）
    - args_schema: Pydantic 参数模型（可选）
    """
    
    # 重写 _run，兼容 LangChain 可能传入的 *args 和 **kwargs
    #统一的入口 所有工具都通过run调用 不管别人传多少个参数、什么名字的参数，我全部收下
    def _run(self, *args: Any, **kwargs: Any) -> str:
        """
        工具调用入口
        LangChain 会根据 args_schema 将参数自动解析为 kwargs，
        但某些情况下也可能传入位置参数，因此这里做了兼容。
        """
        try:
            # 如果子类实现了 _execute，将参数全部传递 #统一异常处理
            return self._execute(*args, **kwargs)
        except Exception as e:
            return f"工具执行出错: {str(e)}"
    #提供异步接口LangChain 如果以异步方式调用工具，就会进入 _arun()
    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """异步执行（默认调用同步方法）"""
        return self._run(*args, **kwargs)
    #强制子类实现业务逻辑 任何继承 BaseTool 的工具都必须实现 _execute()，否则不能实例化。
    @abstractmethod
    def _execute(self, *args, **kwargs) -> str:
        """子类实现具体的工具逻辑"""
        pass