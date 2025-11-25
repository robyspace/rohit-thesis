"""
AWS Lambda Adapter
Implements vendor-neutral interface for AWS Lambda functions
"""

import boto3
import json
import time
import base64
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .base_adapter import (
    CloudAdapter,
    CloudProvider,
    DeploymentConfig,
    FunctionMetadata,
    InvocationResult,
    DeploymentStatus,
    DeploymentError,
    FunctionNotFoundError,
    InvocationError
)


class AWSAdapter(CloudAdapter):
    """AWS Lambda adapter implementing vendor-neutral interface"""

    def __init__(self, region: str = "us-east-1", credentials: Optional[Dict[str, str]] = None):
        super().__init__(region, credentials)

        # Initialize boto3 clients
        session_kwargs = {'region_name': region}
        if credentials:
            session_kwargs.update({
                'aws_access_key_id': credentials.get('access_key_id'),
                'aws_secret_access_key': credentials.get('secret_access_key'),
                'aws_session_token': credentials.get('session_token')
            })

        self.session = boto3.Session(**session_kwargs)
        self._lambda_client = self.session.client('lambda')
        self._logs_client = self.session.client('logs')
        self._cloudwatch_client = self.session.client('cloudwatch')

    @property
    def provider(self) -> CloudProvider:
        return CloudProvider.AWS

    def deploy_function(self, config: DeploymentConfig) -> FunctionMetadata:
        """Deploy function to AWS Lambda"""
        try:
            # Prepare function code
            if config.code_zip:
                code = {'ZipFile': config.code_zip}
            elif config.code_path:
                with open(config.code_path, 'rb') as f:
                    code = {'ZipFile': f.read()}
            else:
                raise DeploymentError("Either code_zip or code_path must be provided")

            # Create function
            response = self._lambda_client.create_function(
                FunctionName=config.function_name,
                Runtime=config.runtime,
                Role=self._get_or_create_execution_role(),
                Handler=config.handler,
                Code=code,
                Description=f"Deployed via vendor-neutral adapter",
                Timeout=config.timeout_seconds,
                MemorySize=config.memory_mb,
                Environment={'Variables': config.environment_variables or {}},
                Tags=config.tags or {},
                Layers=config.layers or []
            )

            return self._response_to_metadata(response)

        except self._lambda_client.exceptions.ResourceConflictException:
            raise DeploymentError(f"Function {config.function_name} already exists")
        except Exception as e:
            raise DeploymentError(f"Deployment failed: {str(e)}")

    def update_function(self, function_name: str, config: DeploymentConfig) -> FunctionMetadata:
        """Update existing Lambda function"""
        try:
            # Update code if provided
            if config.code_zip or config.code_path:
                if config.code_zip:
                    code = config.code_zip
                else:
                    with open(config.code_path, 'rb') as f:
                        code = f.read()

                self._lambda_client.update_function_code(
                    FunctionName=function_name,
                    ZipFile=code
                )

            # Update configuration
            response = self._lambda_client.update_function_configuration(
                FunctionName=function_name,
                Runtime=config.runtime,
                Handler=config.handler,
                Timeout=config.timeout_seconds,
                MemorySize=config.memory_mb,
                Environment={'Variables': config.environment_variables or {}}
            )

            return self._response_to_metadata(response)

        except self._lambda_client.exceptions.ResourceNotFoundException:
            raise FunctionNotFoundError(f"Function {function_name} not found")
        except Exception as e:
            raise DeploymentError(f"Update failed: {str(e)}")

    def delete_function(self, function_name: str) -> bool:
        """Delete Lambda function"""
        try:
            self._lambda_client.delete_function(FunctionName=function_name)
            return True
        except self._lambda_client.exceptions.ResourceNotFoundException:
            return False
        except Exception:
            return False

    def get_function(self, function_name: str) -> FunctionMetadata:
        """Get Lambda function metadata"""
        try:
            response = self._lambda_client.get_function(FunctionName=function_name)
            return self._response_to_metadata(response['Configuration'])
        except self._lambda_client.exceptions.ResourceNotFoundException:
            raise FunctionNotFoundError(f"Function {function_name} not found")

    def list_functions(self) -> List[FunctionMetadata]:
        """List all Lambda functions in region"""
        functions = []
        paginator = self._lambda_client.get_paginator('list_functions')

        for page in paginator.paginate():
            for func in page['Functions']:
                functions.append(self._response_to_metadata(func))

        return functions

    def invoke_function(
        self,
        function_name: str,
        payload: Optional[Dict[str, Any]] = None,
        synchronous: bool = True
    ) -> InvocationResult:
        """Invoke Lambda function"""
        try:
            invocation_type = 'RequestResponse' if synchronous else 'Event'
            payload_bytes = json.dumps(payload or {}).encode('utf-8')

            start_time = time.time()
            response = self._lambda_client.invoke(
                FunctionName=function_name,
                InvocationType=invocation_type,
                Payload=payload_bytes
            )
            execution_duration = (time.time() - start_time) * 1000  # ms

            # Parse response
            status_code = response['StatusCode']
            success = status_code == 200

            response_payload = None
            error_message = None

            if 'Payload' in response:
                payload_str = response['Payload'].read().decode('utf-8')
                try:
                    response_payload = json.loads(payload_str)
                except:
                    response_payload = {'raw': payload_str}

            if 'FunctionError' in response:
                success = False
                error_message = response.get('FunctionError')

            return InvocationResult(
                success=success,
                status_code=status_code,
                response_payload=response_payload,
                execution_duration_ms=execution_duration,
                billed_duration_ms=int(execution_duration),  # Approximation
                memory_used_mb=0,  # Not available in response
                error_message=error_message
            )

        except Exception as e:
            raise InvocationError(f"Invocation failed: {str(e)}")

    def get_function_metrics(
        self,
        function_name: str,
        start_time: str,
        end_time: str
    ) -> Dict[str, Any]:
        """Get CloudWatch metrics for Lambda function"""
        try:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))

            metrics = {}

            # Get invocations
            invocations = self._cloudwatch_client.get_metric_statistics(
                Namespace='AWS/Lambda',
                MetricName='Invocations',
                Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
                StartTime=start_dt,
                EndTime=end_dt,
                Period=300,
                Statistics=['Sum']
            )
            metrics['invocations'] = sum([dp['Sum'] for dp in invocations['Datapoints']])

            # Get errors
            errors = self._cloudwatch_client.get_metric_statistics(
                Namespace='AWS/Lambda',
                MetricName='Errors',
                Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
                StartTime=start_dt,
                EndTime=end_dt,
                Period=300,
                Statistics=['Sum']
            )
            metrics['errors'] = sum([dp['Sum'] for dp in errors['Datapoints']])

            # Get duration
            duration = self._cloudwatch_client.get_metric_statistics(
                Namespace='AWS/Lambda',
                MetricName='Duration',
                Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
                StartTime=start_dt,
                EndTime=end_dt,
                Period=300,
                Statistics=['Average', 'Maximum']
            )
            if duration['Datapoints']:
                metrics['avg_duration_ms'] = sum([dp['Average'] for dp in duration['Datapoints']]) / len(duration['Datapoints'])
                metrics['max_duration_ms'] = max([dp['Maximum'] for dp in duration['Datapoints']])

            return metrics

        except Exception as e:
            return {'error': str(e)}

    def update_function_config(
        self,
        function_name: str,
        memory_mb: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        environment_variables: Optional[Dict[str, str]] = None
    ) -> FunctionMetadata:
        """Update Lambda function configuration"""
        try:
            kwargs = {'FunctionName': function_name}

            if memory_mb is not None:
                kwargs['MemorySize'] = memory_mb
            if timeout_seconds is not None:
                kwargs['Timeout'] = timeout_seconds
            if environment_variables is not None:
                kwargs['Environment'] = {'Variables': environment_variables}

            response = self._lambda_client.update_function_configuration(**kwargs)
            return self._response_to_metadata(response)

        except self._lambda_client.exceptions.ResourceNotFoundException:
            raise FunctionNotFoundError(f"Function {function_name} not found")

    def get_function_logs(
        self,
        function_name: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get CloudWatch logs for Lambda function"""
        try:
            log_group = f"/aws/lambda/{function_name}"

            kwargs = {
                'logGroupName': log_group,
                'limit': limit,
                'interleaved': True
            }

            if start_time:
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                kwargs['startTime'] = int(start_dt.timestamp() * 1000)
            if end_time:
                end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                kwargs['endTime'] = int(end_dt.timestamp() * 1000)

            response = self._logs_client.filter_log_events(**kwargs)

            logs = []
            for event in response.get('events', []):
                logs.append({
                    'timestamp': datetime.fromtimestamp(event['timestamp'] / 1000).isoformat(),
                    'message': event['message'],
                    'stream': event.get('logStreamName')
                })

            return logs

        except Exception as e:
            return [{'error': str(e)}]

    def estimate_cost(
        self,
        memory_mb: int,
        execution_time_ms: float,
        invocations: int
    ) -> float:
        """Estimate AWS Lambda cost"""
        # AWS Lambda pricing (as of 2024, us-east-1)
        # $0.20 per 1M requests
        # $0.0000166667 per GB-second

        request_cost = (invocations / 1_000_000) * 0.20

        memory_gb = memory_mb / 1024
        execution_seconds = execution_time_ms / 1000
        gb_seconds = memory_gb * execution_seconds * invocations
        compute_cost = gb_seconds * 0.0000166667

        return request_cost + compute_cost

    def _response_to_metadata(self, response: Dict) -> FunctionMetadata:
        """Convert Lambda API response to FunctionMetadata"""
        return FunctionMetadata(
            function_name=response['FunctionName'],
            function_arn=response['FunctionArn'],
            cloud_provider=CloudProvider.AWS,
            region=self.region,
            memory_mb=response['MemorySize'],
            timeout_seconds=response['Timeout'],
            status=self._parse_status(response.get('State', 'Active')),
            last_modified=response['LastModified'],
            runtime=response['Runtime'],
            code_size_bytes=response['CodeSize'],
            version=response.get('Version')
        )

    def _parse_status(self, state: str) -> DeploymentStatus:
        """Convert Lambda state to DeploymentStatus"""
        status_map = {
            'Pending': DeploymentStatus.PENDING,
            'Active': DeploymentStatus.ACTIVE,
            'Inactive': DeploymentStatus.FAILED,
            'Failed': DeploymentStatus.FAILED
        }
        return status_map.get(state, DeploymentStatus.ACTIVE)

    def _get_or_create_execution_role(self) -> str:
        """Get or create Lambda execution role ARN"""
        # For simplicity, return a default role ARN
        # In production, this would create/retrieve an actual IAM role
        account_id = boto3.client('sts').get_caller_identity()['Account']
        return f"arn:aws:iam::{account_id}:role/lambda-execution-role"
