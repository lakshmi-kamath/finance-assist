from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a task for an agent"""
    id: str
    type: str
    parameters: Dict[str, Any]
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Standard result format for agent operations"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        self.status = AgentStatus.IDLE
        self.last_execution = None
        self.metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'last_error': None
        }
        
        # Agent capabilities
        self.capabilities = self._define_capabilities()
        self.dependencies = self._define_dependencies()
        
        # Initialize agent-specific components
        self._initialize()
    
    @abstractmethod
    def _define_capabilities(self) -> List[str]:
        """Define what this agent can do"""
        pass
    
    def _define_dependencies(self) -> List[str]:
        """Define what this agent depends on (other agents, services, etc.)"""
        return []
    
    def _initialize(self):
        """Initialize agent-specific components"""
        pass
    
    @abstractmethod
    async def execute_task(self, task: Task) -> AgentResult:
        """Execute a specific task"""
        pass
    
    async def execute(self, task_type: str, parameters: Dict[str, Any], **kwargs) -> AgentResult:
        """Main execution method with error handling and metrics"""
        task = Task(
            id=f"{self.agent_id}_{datetime.now().timestamp()}",
            type=task_type,
            parameters=parameters,
            **kwargs
        )
        
        self.status = AgentStatus.RUNNING
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Executing task: {task_type} with params: {parameters}")
            
            # Execute the task
            result = await self.execute_task(task)
            
            # Update metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self.metrics['tasks_completed'] += 1
            self.metrics['total_execution_time'] += execution_time
            self.last_execution = datetime.now()
            
            result.execution_time = execution_time
            self.status = AgentStatus.COMPLETED
            
            self.logger.info(f"Task completed successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            # Handle errors
            execution_time = (datetime.now() - start_time).total_seconds()
            self.metrics['tasks_failed'] += 1
            self.metrics['last_error'] = str(e)
            self.status = AgentStatus.ERROR
            
            self.logger.error(f"Task failed after {execution_time:.2f}s: {e}")
            
            return AgentResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics"""
        return {
            'agent_id': self.agent_id,
            'status': self.status.value,
            'capabilities': self.capabilities,
            'dependencies': self.dependencies,
            'last_execution': self.last_execution.isoformat() if self.last_execution else None,
            'metrics': self.metrics,
            'config': {k: v for k, v in self.config.items() if not k.endswith('_key')}  # Hide sensitive data
        }
    
    def can_handle_task(self, task_type: str) -> bool:
        """Check if this agent can handle a specific task type"""
        return task_type in self.capabilities
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'agent_id': self.agent_id,
            'healthy': True,
            'status': self.status.value,
            'last_execution': self.last_execution.isoformat() if self.last_execution else None,
            'error_rate': self._calculate_error_rate()
        }
    
    def reset_metrics(self):
        """Reset agent metrics"""
        self.metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'last_error': None
        }
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate percentage"""
        total_tasks = self.metrics['tasks_completed'] + self.metrics['tasks_failed']
        if total_tasks == 0:
            return 0.0
        return (self.metrics['tasks_failed'] / total_tasks) * 100
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, status={self.status.value})"
    
    def __repr__(self) -> str:
        return self.__str__()