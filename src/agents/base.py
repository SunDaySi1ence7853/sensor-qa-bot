class AgentFailure(Exception):
    """Agent 的显式失败信号：做不了就明说，不许硬编结果。

    任务2 的 Supervisor 依据 agent/reason 决定重试、跳过或出部分报告。
    """

    def __init__(self, agent: str, reason: str):
        super().__init__(f"[{agent}] {reason}")
        self.agent = agent
        self.reason = reason