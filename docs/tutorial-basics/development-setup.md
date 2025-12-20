---
sidebar_position: 5
title: "1.5 Development Environment Setup"
description: "Creating reproducible, containerized development environments for Physical AI research"
---

# 1.5 Development Environment Setup
## Introduction
In Physical AI research, environment reproducibility is not a convenience—it's a scientific necessity. This section establishes the "reproducible workspace" philosophy and provides complete, production-ready configurations for your development environment.

## 1.5.1 The Reproducible Workspace Philosophy
Why Containers Are Non-Optional
The Dependency Hell Problem:

ROS 2 Humble requires: Ubuntu 22.04, Python 3.10
PyTorch with CUDA 11.8 requires: Specific driver versions
Gazebo Fortress requires: Specific OGRE and SDL versions
Your custom code requires: Specific package versions
Without containers, this becomes an unsolvable matrix of conflicting dependencies.

## Scientific Reproducibility:

Peer review demands: Other researchers must be able to reproduce your results exactly

Long-term maintenance: Code must work years later, despite library updates

Team collaboration: All team members need identical environments

## The Three-State Development Model:

<div className="text-center">
  <img 
    src="/img/deepseek_mermaid_20251220_8d465f.png" 
    alt="The Three-State Development Model"
    width="85%"
    className="shadow--md"
    style={{borderRadius: '12px', margin: '30px 0'}}
  />
  <p><strong>Figure 1.5:</strong> The Three-State Development Model</p>
</div>

# Version Pinning Strategy
## 


# versions.yaml - The single source of truth
environment:
  base_image: "ubuntu:22.04"
  ros_distro: "humble"
  python_version: "3.10.6"
  cuda_version: "11.8.0"
  cudnn_version: "8.6.0"
  pytorch_version: "2.0.1"
  torchvision_version: "0.15.2"
  opencv_version: "4.8.0"
  gazebo_version: "fortress"
  ignition_version: "fortress"

build_arguments:
  cmake_version: "3.22.1"
  gcc_version: "11.3.0"
  pip_version: "23.1.2"

test_requirements:
  pytest_version: "7.3.1"
  pytest_cov_version: "4.1.0"
  pytest_asyncio_version: "0.21.0"

#   1.5.2 Docker-Based Development Container
Complete Docker Configuration
### Dockerfile:

# Base image with ROS 2 Humble
FROM osrf/ros:humble-desktop-full as base

# Set timezone to avoid interactive prompts
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Simulation tools
    gazebo-fortress \
    gazebo-fortress-common \
    libgazebo-fortress-dev \
    # Development tools
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    python3-vcstool \
    git \
    wget \
    curl \
    nano \
    htop \
    tmux \
    # Graphics (for GUI tools)
    x11-apps \
    mesa-utils \
    # Networking
    net-tools \
    iputils-ping \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set up Python environment
ENV PYTHONUNBUFFERED=1
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create workspace directory
WORKDIR /workspace

# Copy dependency files
COPY ./requirements.txt ./requirements.txt
COPY ./rosdep.yaml ./rosdep.yaml

# Initialize rosdep
RUN rosdep init && rosdep update

# Install ROS dependencies
RUN rosdep install --from-paths . --ignore-src -r -y

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Set up entrypoint
COPY ./entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Set environment variables
ENV ROS_DOMAIN_ID=42
ENV ROS_LOCALHOST_ONLY=0
ENV DISPLAY=:0

# Source ROS in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /workspace/install/setup.bash" >> ~/.bashrc

ENTRYPOINT ["./entrypoint.sh"]
CMD ["bash"]

## requirements.txt:
# Core AI/ML
torch==2.0.1
torchvision==0.15.2
transformers==4.31.0
datasets==2.13.1
openai==0.27.8
anthropic==0.3.11

# Computer Vision
opencv-python==4.8.0.74
opencv-contrib-python==4.8.0.74
Pillow==9.5.0
scikit-image==0.21.0
scikit-learn==1.3.0

# Robotics specific
rospkg==1.5.0
catkin_pkg==0.5.2
empy==3.3.4
netifaces==0.11.0

# Development tools
jupyter==1.0.0
jupyterlab==4.0.0
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
pandas==2.0.3
numpy==1.24.3
scipy==1.10.1

# Testing and monitoring
pytest==7.3.1
pytest-cov==4.1.0
pytest-asyncio==0.21.0
pytest-timeout==2.1.0
black==23.3.0
flake8==6.0.0
mypy==1.4.0
pre-commit==3.3.3

# Utilities
tqdm==4.65.0
rich==13.4.2
python-dotenv==1.0.0
pyyaml==6.0


# 1.5.3 Validation Test: "Hello Physical World"
Complete Test Suite
### Test Script (scripts/validate_environment.py):

```python
#!/usr/bin/env python3
"""
Physical AI Development Environment Validation Test
Validates all components of the development environment
"""

import subprocess
import sys
import os
import time
import json
import yaml
import docker
from pathlib import Path
from datetime import datetime

class EnvironmentValidator:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {"passed": 0, "failed": 0, "total": 0}
        }
        
    def run_test(self, name, test_func):
        """Run a single test and record results"""
        print(f"\n{'='*60}")
        print(f"Running test: {name}")
        print(f"{'='*60}")
        
        try:
            start_time = time.time()
            result = test_func()
            elapsed = time.time() - start_time
            
            self.results["tests"][name] = {
                "status": "PASSED",
                "message": result,
                "duration": round(elapsed, 2)
            }
            self.results["summary"]["passed"] += 1
            print(f"✅ {name}: PASSED ({elapsed:.2f}s)")
            print(f"   {result}")
            
        except Exception as e:
            self.results["tests"][name] = {
                "status": "FAILED",
                "message": str(e),
                "duration": 0
            }
            self.results["summary"]["failed"] += 1
            print(f"❌ {name}: FAILED")
            print(f"   Error: {e}")
        
        self.results["summary"]["total"] += 1
    
    def test_docker_installation(self):
        """Test Docker installation and version"""
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        return f"Docker installed: {result.stdout.strip()}"
    
    def test_docker_compose(self):
        """Test Docker Compose installation"""
        result = subprocess.run(
            ["docker-compose", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        return f"Docker Compose installed: {result.stdout.strip()}"
    
    def test_containers_running(self):
        """Check if required containers are running"""
        client = docker.from_env()
        containers = client.containers.list()
        
        required_containers = [
            "physical-ai-ros-master",
            "physical-ai-gazebo",
            "physical-ai-llm-api",
            "physical-ai-dev"
        ]
        
        running = []
        for container in containers:
            if container.name in required_containers:
                running.append(container.name)
        
        if len(running) == len(required_containers):
            return f"All containers running: {', '.join(running)}"
        else:
            missing = set(required_containers) - set(running)
            raise Exception(f"Missing containers: {', '.join(missing)}")
    
    def test_ros2_communication(self):
        """Test ROS 2 node communication"""
        # Test ROS 2 CLI
        result = subprocess.run(
            ["docker", "exec", "physical-ai-dev", "ros2", "node", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if "/rosout" in result.stdout:
            return "ROS 2 communication successful"
        else:
            raise Exception("ROS 2 nodes not responding")
    
    def test_gazebo_simulation(self):
        """Test Gazebo simulation launch"""
        # Check if Gazebo is running in its container
        result = subprocess.run(
            ["docker", "exec", "physical-ai-gazebo", "pgrep", "-f", "gzserver"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return "Gazebo simulation server running"
        else:
            raise Exception("Gazebo not running")
    
    def test_llm_api(self):
        """Test LLM API service"""
        import requests
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                return "LLM API service healthy"
            else:
                raise Exception(f"LLM API returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise Exception("LLM API not accessible")
    
    def test_python_environment(self):
        """Test Python packages and versions"""
        test_code = """
import sys
import torch
import ros2
import numpy as np
import opencv2 as cv

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"NumPy: {np.__version__}")
print(f"OpenCV: {cv.__version__}")
"""
        
        result = subprocess.run(
            ["docker", "exec", "physical-ai-dev", "python3", "-c", test_code],
            capture_output=True,
            text=True,
            check=True
        )
        
        return "Python environment validated:\n" + result.stdout
    
    def test_hello_physical_world(self):
        """Main validation test - Hello Physical World"""
        test_script = """
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class HelloWorldNode(Node):
    def __init__(self):
        super().__init__('hello_world_node')
        self.publisher = self.create_publisher(String, '/hello_world', 10)
        self.subscription = self.create_subscription(
            String,
            '/hello_world',
            self.listener_callback,
            10
        )
        self.received = False
        
    def listener_callback(self, msg):
        self.get_logger().info(f'Received: {msg.data}')
        self.received = True
        
    def publish_message(self):
        msg = String()
        msg.data = 'Hello Physical World from Module 1!'
        self.publisher.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')

def main():
    rclpy.init()
    node = HelloWorldNode()
    
    # Publish message
    node.publish_message()
    
    # Wait for message to be received
    start_time = time.time()
    while not node.received and time.time() - start_time < 5:
        rclpy.spin_once(node, timeout_sec=0.1)
    
    if node.received:
        print("SUCCESS: Message sent and received successfully!")
        print("ROS 2 communication is working correctly.")
    else:
        print("ERROR: Message not received within timeout")
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
"""
        
        # Write test script to container
        subprocess.run(
            ["docker", "exec", "physical-ai-dev", "bash", "-c", f"cat > /tmp/test_hello.py << 'EOF'\n{test_script}\nEOF"],
            check=True
        )
        
        # Run test script
        result = subprocess.run(
            ["docker", "exec", "physical-ai-dev", "python3", "/tmp/test_hello.py"],
            capture_output=True,
            text=True
        )
        
        if "SUCCESS" in result.stdout:
            return "Hello Physical World test passed!"
        else:
            raise Exception(f"Test failed:\n{result.stdout}\n{result.stderr}")
    
    def generate_report(self):
        """Generate validation report"""
        report_dir = Path("validation_reports")
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"validation_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Also generate markdown report
        md_file = report_dir / f"validation_report_{timestamp}.md"
        self._generate_markdown_report(md_file)
        
        return str(report_file)
    
    def _generate_markdown_report(self, filename):
        """Generate markdown format report"""
        with open(filename, 'w') as f:
            f.write("# Physical AI Environment Validation Report\n\n")
            f.write(f"**Generated:** {self.results['timestamp']}\n\n")
            
            f.write("## Summary\n")
            f.write(f"- **Total Tests:** {self.results['summary']['total']}\n")
            f.write(f"- **Passed:** {self.results['summary']['passed']}\n")
            f.write(f"- **Failed:** {self.results['summary']['failed']}\n")
            f.write(f"- **Success Rate:** {self.results['summary']['passed']/self.results['summary']['total']*100:.1f}%\n\n")
            
            f.write("## Detailed Results\n\n")
            for test_name, test_result in self.results['tests'].items():
                status_emoji = "✅" if test_result['status'] == 'PASSED' else "❌"
                f.write(f"### {status_emoji} {test_name}\n")
                f.write(f"- **Status:** {test_result['status']}\n")
                f.write(f"- **Duration:** {test_result['duration']}s\n")
                f.write(f"- **Message:** {test_result['message']}\n\n")
            
            f.write("## Recommendations\n\n")
            if self.results['summary']['failed'] == 0:
                f.write("🎉 **All tests passed!** Your environment is ready for Physical AI development.\n")
            else:
                f.write("⚠️ **Some tests failed.** Check the failed tests above and fix the issues.\n")
    
    def run_all_tests(self):
        """Run all validation tests"""
        print("🚀 Starting Physical AI Environment Validation")
        print("="*60)
        
        tests = [
            ("Docker Installation", self.test_docker_installation),
            ("Docker Compose", self.test_docker_compose),
            ("Containers Running", self.test_containers_running),
            ("ROS 2 Communication", self.test_ros2_communication),
            ("Gazebo Simulation", self.test_gazebo_simulation),
            ("LLM API Service", self.test_llm_api),
            ("Python Environment", self.test_python_environment),
            ("Hello Physical World", self.test_hello_physical_world),
        ]
        
        for name, test_func in tests:
            self.run_test(name, test_func)
        
        # Generate report
        report_file = self.generate_report()
        
        # Print final summary
        print(f"\n{'='*60}")
        print("VALIDATION COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Passed: {self.results['summary']['passed']}")
        print(f"❌ Failed: {self.results['summary']['failed']}")
        print(f"📊 Total:  {self.results['summary']['total']}")
        print(f"📄 Report: {report_file}")
        print(f"{'='*60}")
        
        if self.results['summary']['failed'] == 0:
            print("🎉 Your Physical AI development environment is READY!")
            return True
        else:
            print("⚠️ Some tests failed. Check the report for details.")
            return False

def main():
    """Main entry point"""
    validator = EnvironmentValidator()
    success = validator.run_all_tests()
    
    if success:
        # Run the actual Hello Physical World demo
        print("\n" + "="*60)
        print("RUNNING HELLO PHYSICAL WORLD DEMO")
        print("="*60)
        
        demo_script = """
#!/usr/bin/env python3
"""
        # ... (rest of the demo script from earlier)
        
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
```

#     Quick Start Script
### scripts/setup_environment.sh:

#!/bin/bash

# Physical AI Development Environment Setup Script
# One-command setup for Module 1

set -e  # Exit on error

echo "🚀 Setting up Physical AI Development Environment"
echo "=================================================="

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check for .env file
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "⚠️  Please edit .env file with your API keys"
        echo "   Required: OPENAI_API_KEY, ANTHROPIC_API_KEY, HUGGINGFACE_TOKEN"
        read -p "Press Enter after editing .env file..."
    else
        echo "❌ .env.example not found. Creating basic .env..."
# Add your API keys here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
HUGGINGFACE_TOKEN=your_token_here

ROS_DOMAIN_ID=42
DISPLAY=:0
EOF
        echo "⚠️  Please edit .env file with your API keys"
        exit 1
    fi
fi

# Build Docker images
echo "🐳 Building Docker images..."
docker-compose build --no-cache

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Run validation tests
echo "🧪 Running validation tests..."
docker exec physical-ai-dev python3 /workspace/scripts/validate_environment.py

echo "=================================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Connect to development container:"
echo "   docker exec -it physical-ai-dev bash"
echo ""
echo "2. Run Hello Physical World test:"
echo "   python3 scripts/hello_physical_world.py"
echo ""
echo "3. Start developing:"
echo "   cd /workspace"
echo "   code .  # If using VS Code with Docker extension"
echo "=================================================="