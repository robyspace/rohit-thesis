"""
Base Cloud Adapter Interface
Defines the vendor-neutral abstraction for serverless function deployment
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class CloudProvider(Enum):
    """Supported cloud providers"""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"


class DeploymentStatus(Enum):
    """Function deployment status"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    FAILED = "failed"
    UPDATING = "updating"
    DELETING = "deleting"
    DELETED = "deleted"


@dataclass
class DeploymentConfig:
    """Vendor-neutral deployment configuration"""
    function_name: str
    runtime: str  # e.g., "python3.10", "nodejs18"
    memory_mb: int
    timeout_seconds: int
    region: str
    environment_variables: Optional[Dict[str, str]] = None
    handler: str = "handler.main"  # Entry point
    code_path: Optional[str] = None  # Local path to code
    code_zip: Optional[bytes] = None  # Zipped code bytes
    layers: Optional[List[str]] = None  # Dependencies/layers
    vpc_config: Optional[Dict[str, Any]] = None  # VPC configuration
    tags: Optional[Dict[str, str]] = None


@dataclass
class FunctionMetadata:
    """Metadata about deployed function"""
    function_name: str
    function_arn: str
    cloud_provider: CloudProvider
    region: str
    memory_mb: int
    timeout_seconds: int
    status: DeploymentStatus
    last_modified: str
    runtime: str
    code_size_bytes: int
    endpoint_url: Optional[str] = None
    version: Optional[str] = None


@dataclass
class InvocationResult:
    """Result of function invocation"""
    success: bool
    status_code: int
    response_payload: Optional[Dict[str, Any]]
    execution_duration_ms: float
    billed_duration_ms: int
    memory_used_mb: int
    log_output: Optional[str] = None
    error_message: Optional[str] = None


class CloudAdapter(ABC):
    """
    Abstract base class for cloud provider adapters

    Each cloud provider (AWS, Azure, GCP) implements this interface
    to provide vendor-neutral serverless function management.
    """

    def __init__(self, region: str, credentials: Optional[Dict[str, str]] = None):
        """
        Initialize cloud adapter

        Args:
            region: Cloud region (e.g., 'us-east-1', 'eastus', 'us-central1')
            credentials: Optional credentials dict (provider-specific)
        """
        self.region = region
        self.credentials = credentials or {}
        self._client = None

    @property
    @abstractmethod
    def provider(self) -> CloudProvider:
        """Return the cloud provider type"""
        pass

    @abstractmethod
    def deploy_function(self, config: DeploymentConfig) -> FunctionMetadata:
        """
        Deploy a serverless function

        Args:
            config: Deployment configuration

        Returns:
            FunctionMetadata with deployment details

        Raises:
            DeploymentError: If deployment fails
        """
        pass

    @abstractmethod
    def update_function(self, function_name: str, config: DeploymentConfig) -> FunctionMetadata:
        """
        Update an existing function

        Args:
            function_name: Name of function to update
            config: Updated configuration

        Returns:
            Updated FunctionMetadata
        """
        pass

    @abstractmethod
    def delete_function(self, function_name: str) -> bool:
        """
        Delete a serverless function

        Args:
            function_name: Name of function to delete

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def get_function(self, function_name: str) -> FunctionMetadata:
        """
        Get metadata about a deployed function

        Args:
            function_name: Name of function

        Returns:
            FunctionMetadata

        Raises:
            FunctionNotFoundError: If function doesn't exist
        """
        pass

    @abstractmethod
    def list_functions(self) -> List[FunctionMetadata]:
        """
        List all deployed functions in the region

        Returns:
            List of FunctionMetadata
        """
        pass

    @abstractmethod
    def invoke_function(
        self,
        function_name: str,
        payload: Optional[Dict[str, Any]] = None,
        synchronous: bool = True
    ) -> InvocationResult:
        """
        Invoke a serverless function

        Args:
            function_name: Name of function to invoke
            payload: Input payload as dict
            synchronous: If True, wait for response; if False, async invocation

        Returns:
            InvocationResult with execution details
        """
        pass

    @abstractmethod
    def get_function_metrics(
        self,
        function_name: str,
        start_time: str,
        end_time: str
    ) -> Dict[str, Any]:
        """
        Get function execution metrics

        Args:
            function_name: Name of function
            start_time: Start time in ISO format
            end_time: End time in ISO format

        Returns:
            Dict with metrics (invocations, errors, duration, etc.)
        """
        pass

    @abstractmethod
    def update_function_config(
        self,
        function_name: str,
        memory_mb: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        environment_variables: Optional[Dict[str, str]] = None
    ) -> FunctionMetadata:
        """
        Update function configuration without changing code

        Args:
            function_name: Name of function
            memory_mb: New memory allocation
            timeout_seconds: New timeout
            environment_variables: New environment variables

        Returns:
            Updated FunctionMetadata
        """
        pass

    @abstractmethod
    def get_function_logs(
        self,
        function_name: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get function execution logs

        Args:
            function_name: Name of function
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of log entries

        Returns:
            List of log entries
        """
        pass

    def normalize_region(self, region: str) -> str:
        """
        Normalize region name to provider-specific format

        Args:
            region: Generic region name

        Returns:
            Provider-specific region name
        """
        # Default implementation - override in subclasses if needed
        return region

    def estimate_cost(
        self,
        memory_mb: int,
        execution_time_ms: float,
        invocations: int
    ) -> float:
        """
        Estimate cost for function executions

        Args:
            memory_mb: Memory allocation
            execution_time_ms: Average execution time
            invocations: Number of invocations

        Returns:
            Estimated cost in USD
        """
        # Default implementation - override in subclasses with actual pricing
        return 0.0


class DeploymentError(Exception):
    """Raised when deployment fails"""
    pass


class FunctionNotFoundError(Exception):
    """Raised when function doesn't exist"""
    pass


class InvocationError(Exception):
    """Raised when function invocation fails"""
    pass
