---
sidebar_position: 1
title: "Module 1: The Architecture of Embodiment"
description: "Foundational principles of Physical AI, humanoid robotics, and embodied intelligence systems"
slug: /module-1/architecture-of-embodiment
---

# The Architecture of Embodiment

## Module Overview

**Estimated Time:** 140-170 hours  
**Prerequisites:** Python, Linux command line, basic linear algebra  
**Technologies:** ROS 2 Humble, Docker, Gazebo, OpenAI API

### Learning Objectives

By the end of this module, you will be able to:

1. Articulate the fundamental differences between digital AI and Physical AI
2. Identify major humanoid robotics platforms and their design philosophies
3. Justify the simulation-first development paradigm with technical arguments
4. Diagram the three-layer architecture of modern Physical AI systems
5. Establish a reproducible ROS 2 development environment using containers
6. Formalize tasks as Vision-Language-Action (VLA) problems
7. Execute the "Fetch It" baseline project with documented performance
8. Trace technical decisions to foundational academic literature

## Module Structure

This module is organized into eight core sections:

1. **The Paradigm Shift** - From digital to physical intelligence
2. **Humanoid Platforms** - Technical specifications and design philosophies
3. **Simulation-First Mandate** - Digital twins and reality gap
4. **Integrated Technical Stack** - Three-layer architecture and ROS 2
5. **Development Environment** - Reproducible workspace setup
6. **VLA Pipeline** - Vision-Language-Action integration
7. **Project: Fetch It** - Baseline implementation
8. **Academic Framework** - Citations and ethical considerations

## Quick Start

To begin immediately with hands-on implementation:

```bash
# Clone the project repository
git clone https://github.com/physical-ai-research/textbook.git
cd textbook/module-1

# Start the development environment
docker-compose up -d

# Run the validation test
docker exec -it module1-dev ./scripts/hello_physical_world.py