"""
记忆管理器
统一管理短期记忆（会话上下文）和长期记忆（用户偏好）
支持Redis持久化，失败时降级为内存存储

"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging

from storage import RedisClient, MySQLClient
from config.settings import settings

logger = logging.getLogger(__name__)

#记忆模块类
class MemoryManager:
    def __init__(self):
        self.redis = RedisClient()#redis  短期记忆
        self.mysql = MySQLClient()#mysql 持久化记忆
        self._short_term_cache: Dict[str, List[Dict[str, Any]]] = {} #内存里的“临时便签”，Redis 挂了也能撑住。

#添加信息方法
    def add_message(self, session_id: str, role: str, content: str) -> None:#形参是会话ID 角色 内容
        message = {#信息内容包括 角色 内容 时间戳
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        key = f"chat:{session_id}"#key是表明此次聊天的会话id是哪个

        # 设定一下Redis 存储的格式、条件等。
        self.redis.lpush(key, message)#包含会话ID和lpush 表示从 Redis 列表左边插入，也就是把最新消息放到最前面
        self.redis.ltrim(key, 0, settings.SHORT_TERM_MEMORY_MAX_LENGTH - 1)#ltrim 表示只保留指定范围内的元素，删除其他元素，redis中保留每个会话最近 10 条消息。

        # 设定MySQL 持久化储存的格式、条件等
        self.mysql.add_conversation(session_id, role, content)#添加的内容 会话ID、角色、内容

        # 降级缓存：无论 Redis 是否可用，都维护本地列表 兜底机制
        if session_id not in self._short_term_cache:
            self._short_term_cache[session_id] = []#如果当前会话还没有缓存，就先创建一个空列表
        self._short_term_cache[session_id].append(message)#向这个空列表中写入message
        if len(self._short_term_cache[session_id]) > settings.SHORT_TERM_MEMORY_MAX_LENGTH:#如果message中内容的长度大于短期记忆最大保留条数10条
            self._short_term_cache[session_id] = self._short_term_cache[session_id][-settings.SHORT_TERM_MEMORY_MAX_LENGTH:]#就只保留最后 10 条，也就是最新 10 条

#获取信息的方法 按优先级读取会话消息，实现 Redis → 内存缓存 → MySQL 的三级读取
    def get_messages(self, session_id: str, limit: int = None) -> List[Dict[str, Any]]:#形参：会话id limit默认为None ；返回一个列表套字典
        key = f"chat:{session_id}"#定义key
        messages = []#初始化一个空列表

        if self.redis.available:#第一优先级：Redis
            raw_msgs = self.redis.lrange(key, 0, -1)
            messages = list(reversed(raw_msgs))
        elif session_id in self._short_term_cache:#第二优先级：进程内存缓存
            messages = self._short_term_cache[session_id].copy()
        else:#第三优先级：MySQL
            # 从 MySQL 获取并加载到缓存
            db_msgs = self.mysql.get_conversation_history(session_id, limit or 20)#有 limit 就读取指定条数，没有 limit 就默认读取最近 20 条。
            for msg in db_msgs:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg["created_at"].isoformat() if hasattr(msg["created_at"], "isoformat") else str(msg["created_at"])
                })
            self._short_term_cache[session_id] = messages.copy()#并写回内存缓存

        if limit:
            messages = messages[-limit:]
        return messages#目前 get_messages() 主要在每次进入 ReAct 流程前加载会话上下文，以及通过 FastAPI 查询会话历史时调用；Planner 正常路径不会调用，Planner 为空降级到 ReAct 时才会调用。

#获得聊天历史记录 给 LLM 看的文本
    def get_chat_history_string(self, session_id: str, limit: int = 6) -> str:
        messages = self.get_messages(session_id, limit)#这个message是被获取的记忆messages 调用上一个方法 获取会话id和近6轮的对话记录
        if not messages:
            return "（无历史对话）"#如果没有找到历史对话记忆 就返回字符串 没有历史对话
        lines = [f"{'用户' if m['role']=='user' else '助手'}: {m['content']}" for m in messages]
        return "\n".join(lines)#把消息列表转成“用户: ... / 助手: ...”的纯文本，方便塞进 prompt 当历史对话。

#删除会话记录
    def clear_session(self, session_id: str) -> None:#情况记忆
        key = f"chat:{session_id}"
        self.redis.delete(key)
        self._short_term_cache.pop(session_id, None)
        logger.info(f"会话 {session_id} 记忆已清除")
    
    # ========== 长期记忆（用户偏好）==========
#用于保存用户长期偏好，同时写入 Redis 和 MySQL
    def set_preference(self, user_id: str, key: str, value: Any) -> bool:
        """
        设置用户偏好
        
        Args:
            user_id: 用户ID
            key: 偏好键
            value: 偏好值
        
        Returns:
            是否成功
        """
        # 存入Redis
        redis_key = f"{settings.LONG_TERM_MEMORY_PREFIX}{user_id}:{key}"
        self.redis.set(redis_key, value)
        # 存入MySQL
        return self.mysql.set_preference(user_id, key, json.dumps(value, ensure_ascii=False))
#获取用户偏好
    def get_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """
        获取用户偏好
        
        Args:
            user_id: 用户ID
            key: 偏好键
            default: 默认值 None
        
        Returns:
            偏好值
        """
        # 优先从Redis获取
        redis_key = f"{settings.LONG_TERM_MEMORY_PREFIX}{user_id}:{key}"
        value = self.redis.get(redis_key)
        if value is not None:
            return value
        
        # 从MySQL获取
        db_value = self.mysql.get_preference(user_id, key)
        if db_value is not None:
            try:
                return json.loads(db_value)
            except:
                return db_value
        
        return default
    
#    获取用户所有偏好
    def get_all_preferences(self, user_id: str) -> Dict[str, Any]:
        """获取用户所有偏好"""
        prefs = self.mysql.get_all_preferences(user_id)
        result = {}
        for k, v in prefs.items():
            try:
                result[k] = json.loads(v)
            except:
                result[k] = v
        return result
    
    # ========== 记忆摘要（用于上下文压缩）==========
    def generate_summary(self, session_id: str, llm=None) -> str:#记忆摘要
        """
        生成会话摘要（调用LLM总结对话）
        
        Args:
            session_id: 会话ID
            llm: LLM实例
        
        Returns:
            摘要字符串
        """
        messages = self.get_messages(session_id, limit=20)#取最近 20 条对话，让 LLM 总结成摘要，然后以用户偏好形式存下来（session:xxx 的 key）。
        if not messages or not llm:
            return "会话暂无足够内容"
        
        conversation = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt = f"""请对以下对话进行简洁的摘要总结，突出关键信息和用户需求。
对话内容：
{conversation}

摘要："""
        
        try:
            summary = llm.invoke(prompt)
            self.set_preference(f"session:{session_id}", "summary", summary)
            return summary.strip()
        except Exception as e:
            logger.error(f"生成摘要失败: {e}")
            return "摘要生成失败"