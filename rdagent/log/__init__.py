# 导入自定义的日志记录器和颜色工具
from rdagent.log.logger import RDAgentLog
from rdagent.log.utils import LogColors

# 创建一个全局的 RDAgentLog 实例，供整个应用程序使用
rdagent_logger: RDAgentLog = RDAgentLog()
