"""
Cloud Orchestrator
Combines hierarchical DRL decision-making with vendor-neutral adapters
"""

import sys
import os
from pathlib import Path
from typing import Dict, Optional
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from adapters.base_adapter import CloudAdapter, DeploymentConfig, FunctionMetadata, CloudProvider
from adapters.aws_adapter import AWSAdapter
from adapters.azure_adapter import AzureAdapter
from adapters.gcp_adapter import GCPAdapter
from inference.hierarchical_coordinator import HierarchicalCoordinator


class CloudOrchestrator:
    """
    Orchestrates multi-cloud serverless deployments using hierarchical DRL

    Combines:
    1. Hierarchical DRL framework for intelligent placement decisions
    2. Vendor-neutral adapters for cloud-agnostic deployment
    """

    def __init__(
        self,
        strategic_model_path: str,
        tactical_model_path: str,
        operational_model_path: str,
        aws_credentials: Optional[Dict[str, str]] = None,
        azure_credentials: Optional[Dict[str, str]] = None,
        gcp_credentials: Optional[Dict[str, str]] = None
    ):
        """
        Initialize cloud orchestrator

        Args:
            strategic_model_path: Path to DQN model
            tactical_model_path: Path to PPO model
            operational_model_path: Path to LSTM model
            aws_credentials: Optional AWS credentials
            azure_credentials: Optional Azure credentials
            gcp_credentials: Optional GCP credentials
        """
        # Initialize hierarchical DRL coordinator
        self.coordinator = HierarchicalCoordinator(
            strategic_model_path=strategic_model_path,
            tactical_model_path=tactical_model_path,
            operational_model_path=operational_model_path
        )

        # Initialize cloud adapters
        self.adapters = {}

        # AWS adapter (always available with boto3)
        self.adapters[CloudProvider.AWS] = {
            'us-east-1': AWSAdapter('us-east-1', aws_credentials),
            'us-west-2': AWSAdapter('us-west-2', aws_credentials),
            'eu-west-1': AWSAdapter('eu-west-1', aws_credentials),
            'ap-southeast-1': AWSAdapter('ap-southeast-1', aws_credentials)
        }

        # Azure adapter (always available for cost estimation, deployment requires credentials)
        self.adapters[CloudProvider.AZURE] = {
            'us-east-1': AzureAdapter('eastus', azure_credentials),
            'us-west-2': AzureAdapter('westus2', azure_credentials),
            'eu-west-1': AzureAdapter('westeurope', azure_credentials),
            'ap-southeast-1': AzureAdapter('southeastasia', azure_credentials)
        }

        # GCP adapter (always available for cost estimation, deployment requires credentials)
        self.adapters[CloudProvider.GCP] = {
            'us-east-1': GCPAdapter('us-east1', gcp_credentials),
            'us-west-2': GCPAdapter('us-west2', gcp_credentials),
            'eu-west-1': GCPAdapter('europe-west1', gcp_credentials),
            'ap-southeast-1': GCPAdapter('asia-southeast1', gcp_credentials)
        }

        print(f"Cloud Orchestrator initialized")
        print(f"  Available clouds: {list(self.adapters.keys())}")
        print(f"  DRL coordinator: {self.coordinator}")

    def deploy_function_intelligent(
        self,
        config: DeploymentConfig,
        strategic_state: np.ndarray,
        tactical_state: np.ndarray,
        operational_sequence: Optional[np.ndarray],
        app_profile: Dict
    ) -> Dict:
        """
        Deploy function with intelligent placement decision

        Args:
            config: Deployment configuration (cloud-agnostic)
            strategic_state: Strategic features for DRL
            tactical_state: Tactical features for DRL
            operational_sequence: Optional operational sequence
            app_profile: Application profile

        Returns:
            Dict with placement decision and deployment metadata
        """
        # Step 1: Make intelligent placement decision using DRL
        decision = self.coordinator.make_decision(
            strategic_state=strategic_state,
            tactical_state=tactical_state,
            operational_sequence=operational_sequence,
            app_profile=app_profile
        )

        cloud_provider = decision['cloud_provider']
        region = decision['region']
        memory_mb = decision['memory_mb']

        print(f"\nIntelligent Placement Decision:")
        print(f"  Cloud: {cloud_provider}")
        print(f"  Region: {region}")
        print(f"  Memory: {memory_mb} MB")
        print(f"  Confidence: {decision['confidence']}")

        # Step 2: Update deployment config with DRL decision
        config.memory_mb = memory_mb
        config.region = region

        # Step 3: Get appropriate cloud adapter
        cloud_enum = CloudProvider[cloud_provider.upper()]

        if cloud_enum not in self.adapters:
            raise ValueError(f"Cloud provider {cloud_provider} not configured")

        if region not in self.adapters[cloud_enum]:
            raise ValueError(f"Region {region} not available for {cloud_provider}")

        adapter = self.adapters[cloud_enum][region]

        # Step 4: Deploy using vendor-neutral adapter
        print(f"\nDeploying to {cloud_provider}/{region}...")

        try:
            metadata = adapter.deploy_function(config)

            return {
                'success': True,
                'decision': decision,
                'deployment': {
                    'function_name': metadata.function_name,
                    'function_arn': metadata.function_arn,
                    'cloud_provider': metadata.cloud_provider.value,
                    'region': metadata.region,
                    'memory_mb': metadata.memory_mb,
                    'status': metadata.status.value
                },
                'metadata': metadata
            }

        except Exception as e:
            return {
                'success': False,
                'decision': decision,
                'error': str(e)
            }

    def migrate_function(
        self,
        function_name: str,
        source_cloud: str,
        source_region: str,
        target_cloud: str,
        target_region: str,
        config: DeploymentConfig
    ) -> Dict:
        """
        Migrate function between clouds (vendor lock-in mitigation)

        Args:
            function_name: Name of function to migrate
            source_cloud: Source cloud provider
            source_region: Source region
            target_cloud: Target cloud provider
            target_region: Target region
            config: Deployment configuration for target

        Returns:
            Dict with migration result
        """
        print(f"\nMigrating function: {function_name}")
        print(f"  From: {source_cloud}/{source_region}")
        print(f"  To: {target_cloud}/{target_region}")

        try:
            # Step 1: Get function metadata from source
            source_adapter = self._get_adapter(source_cloud, source_region)
            source_metadata = source_adapter.get_function(function_name)

            print(f"  Source function found: {source_metadata.function_arn}")

            # Step 2: Deploy to target cloud
            target_adapter = self._get_adapter(target_cloud, target_region)
            target_metadata = target_adapter.deploy_function(config)

            print(f"  Target function deployed: {target_metadata.function_arn}")

            # Step 3: Optionally delete from source (commented out for safety)
            # source_adapter.delete_function(function_name)

            return {
                'success': True,
                'source': {
                    'cloud': source_cloud,
                    'region': source_region,
                    'arn': source_metadata.function_arn
                },
                'target': {
                    'cloud': target_cloud,
                    'region': target_region,
                    'arn': target_metadata.function_arn
                }
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def list_all_functions(self) -> Dict[str, list]:
        """List functions across all configured clouds"""
        all_functions = {}

        for cloud, regions in self.adapters.items():
            cloud_functions = []

            for region, adapter in regions.items():
                try:
                    functions = adapter.list_functions()
                    cloud_functions.extend(functions)
                except Exception as e:
                    print(f"Error listing {cloud}/{region}: {e}")

            all_functions[cloud.value] = cloud_functions

        return all_functions

    def estimate_multi_cloud_cost(
        self,
        memory_mb: int,
        execution_time_ms: float,
        invocations: int
    ) -> Dict[str, float]:
        """Estimate cost across all clouds"""
        costs = {}

        for cloud, regions in self.adapters.items():
            # Use first region adapter for estimation
            adapter = list(regions.values())[0]
            cost = adapter.estimate_cost(memory_mb, execution_time_ms, invocations)
            costs[cloud.value] = cost

        return costs

    def _get_adapter(self, cloud: str, region: str) -> CloudAdapter:
        """Get cloud adapter for specific cloud and region"""
        cloud_enum = CloudProvider[cloud.upper()]

        if cloud_enum not in self.adapters:
            raise ValueError(f"Cloud provider {cloud} not configured")

        if region not in self.adapters[cloud_enum]:
            raise ValueError(f"Region {region} not available for {cloud}")

        return self.adapters[cloud_enum][region]


if __name__ == "__main__":
    print("Cloud Orchestrator - Vendor-Neutral Multi-Cloud Deployment")
    print("="*70)

    # Example usage (requires trained models)
    model_dir = Path(__file__).parent.parent.parent / "data"

    orchestrator = CloudOrchestrator(
        strategic_model_path=str(model_dir / "best_enhanced_dqn.pt"),
        tactical_model_path=str(model_dir / "best_ppo_tactical.pt"),
        operational_model_path=str(model_dir / "best_lstm_predictor.pt")
    )

    # Example: Estimate costs across clouds
    costs = orchestrator.estimate_multi_cloud_cost(
        memory_mb=512,
        execution_time_ms=150,
        invocations=1_000_000
    )

    print("\nCost Estimation (1M invocations, 512MB, 150ms):")
    for cloud, cost in costs.items():
        print(f"  {cloud}: ${cost:.4f}")
