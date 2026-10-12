"""
企业内部API调用工具
支持OA/CRM系统的流程审批、状态查询等功能
提供模拟模式和真实API模式
"""
import json
import requests
from typing import Optional, Type, Dict, Any, List
from pydantic import BaseModel, Field
import logging
from datetime import datetime

from config.settings import settings
from .base import BaseTool

logger = logging.getLogger(__name__)

#API工具的输入类 APIToolInput 是 APITool 的输入参数模型，用来约束和校验调用 API 工具时必须传入什么参数。
class APIToolInput(BaseModel):
    """API工具输入参数模式"""
    #要执行的操作。必填
    action: str = Field(description="操作类型: process_approval, get_system_status, get_user_info, query_approval_status")
    #目标系统，只能是 OA 或 CRM。必填
    system: str = Field(description="目标系统: OA 或 CRM")
    #审批类型。默认leave
    approval_type: Optional[str] = Field("leave", description="审批类型: leave, expense, contract, purchase 等")
    #用户编号。
    user_id: Optional[str] = Field(None, description="用户ID")
    #附加业务参数。
    params: Optional[Dict[str, Any]] = Field(None, description="额外参数")
    #是否模拟执行。当前实现中 True 表示模拟，False 表示尝试真实 API。
    use_mock: Optional[bool] = Field(False, description="是否使用模拟模式（True=真实API调用，False=模拟数据）")

#ApprovalRecord 是审批记录的数据模型类，用来保存一条模拟审批信息。
class ApprovalRecord:
    """审批记录"""
    def __init__(self, approval_id: str, approval_type: str, system: str, status: str, create_time: str, **kwargs):
        self.approval_id = approval_id
        self.approval_type = approval_type
        self.system = system
        self.status = status
        self.create_time = create_time
        self.extra_data = kwargs
        
#to_dict() 方法把对象转换成字典，方便返回 JSON
    def to_dict(self) -> Dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "approval_type": self.approval_type,
            "system": self.system,
            "status": self.status,
            "create_time": self.create_time,
            **self.extra_data
        }

#企业内部API调用工具
class APITool(BaseTool):
    """
    企业内部API调用工具

    功能：
    1. 发起OA/CRM流程审批（请假、报销、合同、采购等）
    2. 查询系统状态
    3. 获取用户信息
    4. 查询审批状态
    5. 支持模拟模式和真实API模式
    """
    #工具在系统中的唯一名称。Planner 和 Agent 通过这个名字调用它
    name: str = "api_tool"
    #给大模型看的工具说明。ReAct Agent 会读取这段描述，判断什么时候应该使用 api_tool。
    description: str = """
    企业内部系统API调用工具，用于与OA、CRM系统交互。
    支持的操作：
    - process_approval: 发起审批流程，需指定 system 和 approval_type
    - get_system_status: 查询系统运行状态，需指定 system
    - get_user_info: 获取用户信息，需指定 system 和 user_id
    - query_approval_status: 查询审批状态，需指定 approval_id
    支持 use_mock=true 使用模拟数据，use_mock=false 调用真实API。
    """
    #指定工具参数结构。LangChain 在调用工具前，会根据 APIToolInput 校验参数
    args_schema: Type[BaseModel] = APIToolInput
    #保存模拟创建的审批记录
    _mock_approvals: Dict[str, ApprovalRecord] = {}
    #模拟的用户数据
    _mock_users: Dict[str, Dict[str, Dict[str, Any]]] = {
        "OA": {
            "001": {"name": "张三", "department": "人事部", "position": "经理", "email": "zhangsan@company.com"},
            "002": {"name": "李四", "department": "财务部", "position": "主管", "email": "lisi@company.com"},
            "003": {"name": "王五", "department": "技术部", "position": "总监", "email": "wangwu@company.com"},
        },
        "CRM": {
            "001": {"name": "赵六", "department": "销售部", "position": "总监", "email": "zhaoliu@company.com"},
            "002": {"name": "钱七", "department": "市场部", "position": "专员", "email": "qianqi@company.com"},
        }
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)#初始化 LangChain BaseTool
        self._session = requests.Session()#复用 HTTP 连接，而不是每次请求都新建连接。
        self._session.headers.update({"Content-Type": "application/json"})#设置请求头，告知 OA/CRM 服务请求体是 JSON。

#是 API 工具的总入口和模式路由器，负责决定这次请求怎么执行。
    def _execute(self, action: str, system: str, approval_type: Optional[str] = "leave",
                user_id: Optional[str] = None, params: Optional[Dict[str, Any]] = None,
                 use_mock: bool = False, **kwargs) -> str:
        """执行API调用"""
        system = system.upper()#把系统名转成大写 避免 oa、Oa、OA 造成判断差异
        if system not in ["OA", "CRM"]:#只允许 OA 或 CRM
            return json.dumps({"error": f"不支持的系统: {system}，请使用 OA 或 CRM"}, ensure_ascii=False, indent=2)

        if use_mock or not self._is_api_available(system):#选择模拟模式或真实模式 use_mock=True使用模拟模式   False使用真实模式
            return self._mock_execute(action, system, approval_type, user_id, params)#模拟模式
        else:
            return self._real_execute(action, system, approval_type, user_id, params)#真实模式 检查真实 API 是否可用 不可用 → 使用模拟数据   可用   → 调用真实 OA/CRM API

#检查真实API是否可用 _is_api_available() 根据系统读取配置地址，设置 2 秒超时请求 /health，只有返回 200 才认为真实 API 可用；
    def _is_api_available(self, system: str) -> bool:
        """检查真实API是否可用"""
        try:
            if system == "OA":
                base_url = settings.OA_API_BASE_URL
            else:
                base_url = settings.CRM_API_BASE_URL

            if not base_url or base_url == "http://localhost:8000/api/oa":
                return False

            response = self._session.get(f"{base_url}/health", timeout=2)
            return response.status_code == 200 #2秒钟
        except:
            return False#没有配置、超时、连接失败或返回异常状态时都返回 False，系统随后自动降级到模拟模式

#模拟的API调用 _mock_execute() 是模拟 API 的总调度方法，根据 action 分发到发起审批、系统状态、用户信息和审批状态四个具体模拟方法，并统一返回 JSON 字符串。
#当用户明确要求模拟，或者真实 OA/CRM 不可用时，系统会走模拟路线；模拟路线根据操作类型返回审批结果、系统状态、用户信息或审批进度，最终统一输出 JSON 字符串。  
    def _mock_execute(self, action: str, system: str, approval_type: str,
                    user_id: Optional[str], params: Optional[Dict]) -> str:
        """模拟执行API调用"""
        if action == "process_approval":
            return self._mock_process_approval(system, approval_type, params)
        elif action == "get_system_status":
            return self._mock_get_system_status(system)
        elif action == "get_user_info":
            return self._mock_get_user_info(system, user_id)
        elif action == "query_approval_status":
            approval_id = params.get("approval_id") if params else None
            return self._mock_query_approval_status(approval_id)
        else:
            return json.dumps({"error": f"未知操作: {action}，支持: process_approval, get_system_status, get_user_info, query_approval_status"}, ensure_ascii=False, indent=2)

#模拟发起审批，生成审批编号并保存到 _mock_approvals。
    def _mock_process_approval(self, system: str, approval_type: str, params: Optional[Dict]) -> str:
        """模拟发起审批流程"""
        approval_id = f"AP{datetime.now().strftime('%Y%m%d%H%M%S')}{hash(approval_type) % 1000:03d}"

        approval = ApprovalRecord(
            approval_id=approval_id,
            approval_type=approval_type,
            system=system,
            status="pending",
            create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            applicant=params.get("applicant", "当前用户") if params else "当前用户",
            reason=params.get("reason", "未填写") if params else "未填写",
        )

        self._mock_approvals[approval_id] = approval

        logger.info(f"[模拟] 发起{approval_type}审批: {approval_id}")

        result = {
            "status": "success",
            "message": f"已成功在{system}系统发起{approval_type}审批流程",
            "approval_id": approval_id,
            "approval_type": approval_type,
            "system": system,
            "create_time": approval.create_time,
            "estimated_time": "2个工作日",
            "mode": "mock"
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

#返回模拟的 OA/CRM 系统状态。
    def _mock_get_system_status(self, system: str) -> str:
        """模拟获取系统状态"""
        status_data = {
            "OA": {
                "status": "online",
                "version": "2.3.1",
                "uptime": "15天",
                "active_users": 156,
                "pending_approvals": 23,
                "mode": "mock"
            },
            "CRM": {
                "status": "online",
                "version": "4.1.0",
                "uptime": "30天",
                "active_users": 89,
                "pending_tasks": 15,
                "mode": "mock"
            }
        }
        result = status_data.get(system, {})
        result["message"] = f"{system}系统运行正常"
        return json.dumps(result, ensure_ascii=False, indent=2)

#从 _mock_users 查询模拟用户信息。
    def _mock_get_user_info(self, system: str, user_id: Optional[str]) -> str:
        """模拟获取用户信息"""
        if not user_id:
            return json.dumps({"error": "请提供 user_id 参数"}, ensure_ascii=False, indent=2)

        user_info = self._mock_users.get(system, {}).get(user_id)
        if user_info:
            result = {"user_id": user_id, "system": system, **user_info, "mode": "mock"}
            return json.dumps(result, ensure_ascii=False, indent=2)
        else:
            return json.dumps({"error": f"未找到用户 {user_id} 在 {system} 系统中的信息"}, ensure_ascii=False, indent=2)

#根据审批编号，从 _mock_approvals 查询模拟审批状态。
    def _mock_query_approval_status(self, approval_id: Optional[str]) -> str:
        """模拟查询审批状态"""
        if not approval_id:
            return json.dumps({"error": "请提供 approval_id 参数"}, ensure_ascii=False, indent=2)

        approval = self._mock_approvals.get(approval_id)
        if approval:
            result = approval.to_dict()
            result["mode"] = "mock"
            return json.dumps(result, ensure_ascii=False, indent=2)
        else:
            return json.dumps({"error": f"未找到审批记录 {approval_id}"}, ensure_ascii=False, indent=2)

#只有在明确关闭模拟且真实 OA/CRM 健康检查通过时，才会走真实路线。系统会请求 /approvals、/status、/users/{id} 和 /approvals/{id}，并把外部系统响应转换成 JSON 字符串返回。当前项目没有提供完整的真实 OA/CRM 服务，所以默认情况下实际走的仍是模拟路线。
    def _real_execute(self, action: str, system: str, approval_type: str,
                    user_id: Optional[str], params: Optional[Dict]) -> str:
        """真实API执行"""
        base_url = settings.OA_API_BASE_URL if system == "OA" else settings.CRM_API_BASE_URL

        endpoints = {
            "process_approval": f"{base_url}/approvals",
            "get_system_status": f"{base_url}/status",
            "get_user_info": f"{base_url}/users/{user_id}" if user_id else None,
            "query_approval_status": f"{base_url}/approvals/{params.get('approval_id') if params else None}",
        }

        try:
            if action == "process_approval":
                payload = {
                    "approval_type": approval_type,
                    **(params or {})
                }
                response = self._session.post(endpoints["process_approval"], json=payload, timeout=10)
                response.raise_for_status()
                return json.dumps(response.json(), ensure_ascii=False, indent=2)

            elif action == "get_system_status":
                response = self._session.get(endpoints["get_system_status"], timeout=5)
                response.raise_for_status()
                return json.dumps(response.json(), ensure_ascii=False, indent=2)

            elif action == "get_user_info":
                if not user_id:
                    return json.dumps({"error": "请提供 user_id"}, ensure_ascii=False, indent=2)
                response = self._session.get(endpoints["get_user_info"], timeout=5)
                response.raise_for_status()
                return json.dumps(response.json(), ensure_ascii=False, indent=2)

            elif action == "query_approval_status":
                approval_id = params.get("approval_id") if params else None
                if not approval_id:
                    return json.dumps({"error": "请提供 approval_id"}, ensure_ascii=False, indent=2)
                response = self._session.get(f"{base_url}/approvals/{approval_id}", timeout=5)
                response.raise_for_status()
                return json.dumps(response.json(), ensure_ascii=False, indent=2)

            else:
                return json.dumps({"error": f"未知操作: {action}"}, ensure_ascii=False, indent=2)

        except requests.exceptions.RequestException as e:
            logger.error(f"API调用失败: {e}")
            fallback_result = {
                "error": f"API调用失败: {str(e)}",
                "fallback": "已自动降级为模拟模式",
                "suggestion": f"请检查{system}系统服务是否启动，或使用 use_mock=true 使用模拟数据"
            }
            return json.dumps(fallback_result, ensure_ascii=False, indent=2)

    def list_approval_types(self, system: str) -> List[str]:
        """获取支持的审批类型"""
        common_types = ["leave", "expense", "contract", "purchase", "travel", "overtime"]
        if system.upper() == "OA":
            return common_types + ["resign", "training"]
        else:
            return common_types + ["contract_sign", "customer_visit"]
