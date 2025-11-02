"""
这是一个试图存储/恢复/回溯工作流会话的类。

后记:
- 最初，我想用 Python 生成器 (generator) 以更通用的方式实现它。
- 但是，Python 生成器是不可序列化 (pickle) 的（dill 也不支持）。
"""

import asyncio
import concurrent.futures
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union, cast

import psutil
from tqdm.auto import tqdm

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.log import rdagent_logger as logger
from rdagent.log.conf import LOG_SETTINGS
from rdagent.log.timer import RD_Agent_TIMER_wrapper, RDAgentTimer
from rdagent.utils.workflow.tracking import WorkflowTracker


class LoopMeta(type):
    """
    一个元类，用于自动收集一个类及其所有基类中定义的“步骤”（step）方法。
    步骤被定义为不以下划线开头、可调用且非类类型的属性。
    """

    @staticmethod
    def _get_steps(bases: tuple[type, ...]) -> list[str]:
        """
        递归地从所有基类中获取 `steps` 列表，并将它们合并成一个列表。

        Args:
            bases (tuple): 基类的元组。

        Returns:
            List[str]: 从所有基类合并而来的步骤列表。
        """
        steps = []
        for base in bases:
            # 递归调用以获取父类的步骤，然后添加当前基类的步骤
            for step in LoopMeta._get_steps(base.__bases__) + getattr(base, "steps", []):
                if step not in steps and step not in ["load", "dump"]:  # 避免重复和覆盖内置方法
                    steps.append(step)
        return steps

    def __new__(mcs, clsname: str, bases: tuple[type, ...], attrs: dict[str, Any]) -> Any:
        """
        创建一个新类，并将基类和当前类中的步骤合并。
        """
        steps = LoopMeta._get_steps(bases)  # 获取所有父类的步骤
        for name, attr in attrs.items():
            if not name.startswith("_") and callable(attr) and not isinstance(attr, type):
                # 排除私有方法、非可调用对象和类本身
                if name not in steps and name not in ["load", "dump"]:
                    # 如果子类中覆盖了步骤，它已经存在于 `steps` 中，所以这里只添加新步骤
                    steps.append(name)
        attrs["steps"] = steps  # 将收集到的步骤列表存入类属性
        return super().__new__(mcs, clsname, bases, attrs)


@dataclass
class LoopTrace:
    """
    用于记录单个步骤执行轨迹的数据类。
    """
    start: datetime  # 步骤开始时间
    end: datetime  # 步骤结束时间
    step_idx: int  # 步骤索引


class LoopBase:
    """
    工作流循环的基类。

    假设:
    - 最后一个步骤负责记录信息！

    未解决的问题:
    - 当 `force_subproc` 为 True 时，全局变量（如 Timer）的同步问题。
    """

    steps: list[str]  # 要执行的步骤名称列表，由 LoopMeta 自动填充
    loop_trace: dict[int, list[LoopTrace]]  # 记录每个循环的执行轨迹

    # 用户可以定义一组异常类型，当这些异常发生时，当前循环将被跳过
    skip_loop_error: tuple[type[BaseException], ...] = ()
    # 用户可以定义一组异常类型，当这些异常发生时，当前循环将被撤销，并回滚到上一个循环的状态
    withdraw_loop_error: tuple[type[BaseException], ...] = ()

    # 用于在 loop_prev_out 字典中存储特殊信息的键
    EXCEPTION_KEY = "_EXCEPTION"
    LOOP_IDX_KEY = "_LOOP_IDX"
    SENTINEL = -1  # 用于标记队列结束的哨兵值

    _pbar: tqdm  # tqdm 进度条实例

    class LoopTerminationError(Exception):
        """当循环条件表明应终止循环时引发的异常。"""

    class LoopResumeError(Exception):
        """当循环条件表明应停止所有协程并恢复时引发的异常。"""

    def __init__(self) -> None:
        # 进度控制
        self.loop_idx: int = 0  # 当前/下一个要启动的循环索引
        self.step_idx: defaultdict[int, int] = defaultdict(int)  # 映射: 循环索引 -> 下一个要执行的步骤索引
        self.queue: asyncio.Queue[Any] = asyncio.Queue()  # 用于在生产者和消费者之间传递循环任务的队列

        # 存储所有循环的步骤结果
        # 结构: loop_prev_out[循环索引][步骤名称] = 步骤输出
        self.loop_prev_out: dict[int, dict[str, Any]] = defaultdict(dict)
        self.loop_trace = defaultdict(list[LoopTrace])
        self.session_folder = Path(LOG_SETTINGS.trace_path) / "__session__"  # 会话快照存储目录
        self.timer: RDAgentTimer = RD_Agent_TIMER_wrapper.timer  # 全局计时器
        self.tracker = WorkflowTracker(self)  # 工作流追踪器

        # 循环和步骤计数限制
        self.loop_n: Optional[int] = None
        self.step_n: Optional[int] = None

        # 用于控制并发的信号量
        self.semaphores: dict[str, asyncio.Semaphore] = {}

    def get_semaphore(self, step_name: str) -> asyncio.Semaphore:
        """
        获取或创建指定步骤的信号量，用于控制并发执行数。
        """
        if isinstance(limit := RD_AGENT_SETTINGS.step_semaphore, dict):
            limit = limit.get(step_name, 1)  # 如果未指定，默认为 1

        # 特殊处理：record 和 feedback 步骤强制串行执行，以避免竞争条件
        if step_name in ("record", "feedback"):
            limit = 1

        if step_name not in self.semaphores:
            self.semaphores[step_name] = asyncio.Semaphore(limit)
        return self.semaphores[step_name]

    @property
    def pbar(self) -> tqdm:
        """按需初始化的进度条属性。"""
        if getattr(self, "_pbar", None) is None:
            self._pbar = tqdm(total=len(self.steps), desc="工作流进度", unit="step")
        return self._pbar

    def close_pbar(self) -> None:
        """关闭进度条。"""
        if getattr(self, "_pbar", None) is not None:
            self._pbar.close()
            del self._pbar

    def _check_exit_conditions_on_step(self, loop_id: Optional[int] = None, step_id: Optional[int] = None) -> None:
        """
        检查循环是否应继续或终止。
        """
        # 检查步骤数限制
        if self.step_n is not None:
            if self.step_n <= 0:
                raise self.LoopTerminationError("已达到步骤数限制")
            self.step_n -= 1

        # 检查计时器是否超时
        if self.timer.started and self.timer.is_timeout():
            logger.warning("超时，退出循环。")
            raise self.LoopTerminationError("计时器超时")

    async def _run_step(self, li: int, force_subproc: bool = False) -> None:
        """
        异步执行工作流中的单个步骤。

        Parameters
        ----------
        li : int
            循环索引。
        force_subproc : bool
            是否强制在子进程中运行该步骤。
        """
        si = self.step_idx[li]
        name = self.steps[si]

        async with self.get_semaphore(name):  # 获取信号量以控制并发
            logger.info(f"开始循环 {li}, 步骤 {si}: {name}")
            self.tracker.log_workflow_state()

            with logger.tag(f"Loop_{li}.{name}"):
                start = datetime.now(timezone.utc)
                func: Callable[..., Any] = cast(Callable[..., Any], getattr(self, name))
                next_step_idx = si + 1
                step_forward = True
                self.loop_prev_out[li][self.LOOP_IDX_KEY] = li  # 注入当前循环索引

                try:
                    # 根据函数类型（同步/异步）和 force_subproc 标志来执行步骤
                    if force_subproc:
                        curr_loop = asyncio.get_running_loop()
                        with concurrent.futures.ProcessPoolExecutor() as pool:
                            result = await curr_loop.run_in_executor(pool, func, self.loop_prev_out[li])
                    elif asyncio.iscoroutinefunction(func):
                        result = await func(self.loop_prev_out[li])
                    else:
                        result = func(self.loop_prev_out[li])
                    self.loop_prev_out[li][name] = result
                except Exception as e:
                    # 异常处理
                    if isinstance(e, self.skip_loop_error):
                        logger.warning(f"因 {e} 跳过循环 {li}")
                        next_step_idx = len(self.steps) - 1  # 跳转到最后一步
                        self.loop_prev_out[li][name] = None
                        self.loop_prev_out[li][self.EXCEPTION_KEY] = e
                    elif isinstance(e, self.withdraw_loop_error):
                        logger.warning(f"因 {e} 撤销循环 {li}")
                        self.withdraw_loop(li)  # 回滚状态
                        step_forward = False
                        raise self.LoopResumeError("已重置循环实例，停止所有协程并恢复。") from e
                    else:
                        raise  # 重新抛出未处理的异常
                finally:
                    # 记录轨迹并更新状态
                    end = datetime.now(timezone.utc)
                    self.loop_trace[li].append(LoopTrace(start, end, step_idx=si))

                    if step_forward:
                        self.step_idx[li] = next_step_idx
                        self.pbar.n = self.step_idx[li] # 更新进度条
                        # 执行成功后，保存快照
                        if name in self.loop_prev_out[li]:
                            self.dump(self.session_folder / f"{li}" / f"{si}_{name}")
                        self._check_exit_conditions_on_step(loop_id=li, step_id=si)

    async def kickoff_loop(self) -> None:
        """
        生产者协程：启动新的循环并将其放入队列。
        """
        while True:
            li = self.loop_idx
            # 检查循环数限制
            if self.loop_n is not None and self.loop_n <= 0:
                for _ in range(RD_AGENT_SETTINGS.get_max_parallel()):
                    self.queue.put_nowait(self.SENTINEL)
                break

            if self.loop_n is not None:
                self.loop_n -= 1

            # 仅当一个循环从未开始时，才执行其第一步（通常是生成任务）
            if self.step_idx[li] == 0:
                await self._run_step(li)
            self.queue.put_nowait(li)  # 将任务放入队列
            self.loop_idx += 1
            await asyncio.sleep(0)

    async def execute_loop(self) -> None:
        """
        消费者协程：从队列中获取循环任务并执行其未完成的步骤。
        """
        while True:
            li = await self.queue.get()
            if li == self.SENTINEL:
                break
            # 循环执行该任务的所有剩余步骤
            while self.step_idx[li] < len(self.steps):
                await self._run_step(li, force_subproc=RD_AGENT_SETTINGS.is_force_subproc())

    async def run(self, step_n: int | None = None, loop_n: int | None = None, all_duration: str | None = None) -> None:
        """
        运行整个工作流循环。
        """
        if all_duration is not None and not self.timer.started:
            self.timer.reset(all_duration=all_duration)
        self.step_n = step_n
        self.loop_n = loop_n

        # 清空队列并重置循环索引
        while not self.queue.empty():
            self.queue.get_nowait()
        self.loop_idx = 0

        tasks: list[asyncio.Task] = []
        while True:
            try:
                # 创建生产者和消费者任务
                tasks = [
                    asyncio.create_task(self.kickoff_loop()),
                    *[asyncio.create_task(self.execute_loop()) for _ in range(RD_AGENT_SETTINGS.get_max_parallel())],
                ]
                await asyncio.gather(*tasks)
                break
            except self.LoopResumeError as e:
                logger.warning(f"停止所有协程并恢复循环: {e}")
                self.loop_idx = 0
            except self.LoopTerminationError as e:
                logger.warning(f"达到停止标准，停止循环: {e}")
                kill_subprocesses()  # 手动终止子进程
                break
            finally:
                # 在恢复或退出前取消所有任务
                for t in tasks:
                    t.cancel()
                self.close_pbar()

    def withdraw_loop(self, loop_idx: int) -> None:
        """
        撤销一个循环，通过加载上一个循环的快照来恢复状态。
        """
        prev_session_dir = self.session_folder / str(loop_idx - 1)
        # 找到上一个循环的第一个快照
        prev_path = min(
            (p for p in prev_session_dir.glob("*_*") if p.is_file()),
            key=lambda item: int(item.name.split("_", 1)[0]),
            default=None,
        )
        if prev_path:
            loaded = type(self).load(prev_path, checkout=True, replace_timer=True)
            logger.info(f"从 {prev_path} 加载了上一个会话")
            # 用加载的状态覆盖当前实例的状态
            self.__dict__ = loaded.__dict__
        else:
            logger.error(f"在 {prev_session_dir} 中找不到上一个转储，无法撤销循环 {loop_idx}")
            raise RuntimeError("无法撤销循环")

    def dump(self, path: str | Path) -> None:
        """将当前工作流状态序列化 (pickle) 到文件。"""
        if self.timer.started:
            self.timer.update_remain_time()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    def truncate_session_folder(self, li: int, si: int) -> None:
        """
        通过删除给定循环索引 (li) 和步骤索引 (si) 之后的所有会话对象来清理会话文件夹。
        """
        # 清理 li 之后的所有循环目录
        for sf in self.session_folder.iterdir():
            if sf.is_dir() and int(sf.name) > li:
                for file in sf.iterdir():
                    file.unlink()
                sf.rmdir()

        # 清理 li 循环中 si 之后的所有步骤快照
        final_loop_session_folder = self.session_folder / str(li)
        for step_session in final_loop_session_folder.glob("*_*"):
            if step_session.is_file():
                step_id = int(step_session.name.split("_", 1)[0])
                if step_id > si:
                    step_session.unlink()

    @classmethod
    def load(cls, path: str | Path, checkout: bool = False, replace_timer: bool = True) -> "LoopBase":
        """
        从给定路径加载会话。
        """
        path = Path(path)
        # 如果路径是目录，则加载最新的会话文件
        if path.is_dir():
            if not path.exists():
                raise FileNotFoundError(f"在 {path} 中找不到会话文件")
            files = sorted(path.glob("*/*_*"), key=lambda f: (int(f.parent.name), int(f.name.split("_")[0])))
            if not files:
                 raise FileNotFoundError(f"在 {path} 中找不到会话文件")
            path = files[-1]
            logger.info(f"从 {path} 加载最新会话")

        with path.open("rb") as f:
            session = cast(LoopBase, pickle.load(f))

        if checkout:
            # 清理加载点之后的所有日志和会话
            max_loop = max(session.loop_trace.keys())
            session.truncate_session_folder(max_loop, len(session.loop_trace[max_loop]) - 1)
            logger.truncate_storages(session.loop_trace[max_loop][-1].end)

        if session.timer.started and replace_timer:
            RD_Agent_TIMER_wrapper.replace_timer(session.timer)
            RD_Agent_TIMER_wrapper.timer.restart_by_remain_time()

        return session

    def __getstate__(self) -> dict[str, Any]:
        """自定义序列化，排除不可序列化的属性。"""
        return {k: v for k, v in self.__dict__.items() if k not in ["queue", "semaphores", "_pbar"]}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """自定义反序列化，重新初始化不可序列化的属性。"""
        self.__dict__.update(state)
        self.queue = asyncio.Queue()
        self.semaphores = {}


def kill_subprocesses() -> None:
    """
    由于工作流的协程特性，主进程的事件循环无法自动停止由 `run_in_executor` 启动的子进程。
    因此，我们需要手动杀死它们。否则，子进程会继续在后台运行，导致主进程无法退出。
    """
    current_proc = psutil.Process(os.getpid())
    children = current_proc.children(recursive=True)
    for child in children:
        try:
            logger.warning(f"正在终止子进程 PID {child.pid} ({child.name()})")
            child.terminate()
        except psutil.NoSuchProcess:
            pass

    # 等待进程终止，然后强制杀死仍然存活的进程
    _, alive = psutil.wait_procs(children, timeout=3)
    for p in alive:
        try:
            logger.warning(f"正在强制杀死仍然存活的子进程 PID {p.pid} ({p.name()})")
            p.kill()
        except psutil.NoSuchProcess:
            pass
    logger.info("完成子进程清理。")
