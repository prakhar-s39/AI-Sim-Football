#!/usr/bin/env python3
"""
Test script for MAPPO implementation
This script verifies that all components work correctly
"""

import sys
import os
import torch
import numpy as np

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        from football_ai import MAPPOAgent, ParallelSimulationManager, GameLogger
        from game import FootballEnv
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_device_detection():
    """Test GPU device detection"""
    print("Testing device detection...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    return True

def test_mappo_agent():
    """Test MAPPO agent creation and basic functionality"""
    print("Testing MAPPO agent...")
    try:
        from football_ai import MAPPOAgent
        
        # Create agent
        agent = MAPPOAgent()
        print("✓ MAPPO agent created successfully")
        
        # Test action selection
        obs = np.random.randn(12).astype(np.float32)
        action, log_prob, value = agent.act(obs)
        
        print(f"✓ Action: {action}, Log prob: {log_prob.item():.4f}, Value: {value.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ MAPPO agent test failed: {e}")
        return False

def test_environment():
    """Test football environment"""
    print("Testing football environment...")
    try:
        from game import FootballEnv
        
        env = FootballEnv()
        obs = env.reset()
        print(f"✓ Environment created, observation shape: {obs.shape}")
        
        # Test step
        actions = [0, 1]  # Random actions
        obs, rewards, done, info = env.step(actions)
        print(f"✓ Environment step successful, rewards: {rewards}")
        
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_logger():
    """Test game logger"""
    print("Testing game logger...")
    try:
        from football_ai import GameLogger
        
        logger = GameLogger("test_logs")
        obs = np.random.randn(12).astype(np.float32)
        actions = [0, 1]
        rewards = [0, 0]
        
        logger.log_step(obs, actions, rewards, False)
        logger.log_game_end({"left": 1, "right": 0}, 0)
        
        print("✓ Game logger working")
        return True
    except Exception as e:
        print(f"✗ Logger test failed: {e}")
        return False

def test_parallel_manager():
    """Test parallel simulation manager"""
    print("Testing parallel simulation manager...")
    try:
        from football_ai import ParallelSimulationManager
        
        manager = ParallelSimulationManager(num_simulations=2, log_dir="test_logs")
        print("✓ Parallel simulation manager created")
        
        return True
    except Exception as e:
        print(f"✗ Parallel manager test failed: {e}")
        return False

def run_quick_training_test():
    """Run a very quick training test"""
    print("Running quick training test...")
    try:
        from football_ai import MAPPOAgent
        from game import FootballEnv
        
        # Create environment and agents
        env = FootballEnv()
        agent1 = MAPPOAgent()
        agent2 = MAPPOAgent()
        
        # Run a few steps
        obs = env.reset()
        for step in range(10):
            action1, log_prob1, value1 = agent1.act(obs)
            action2, log_prob2, value2 = agent2.act(obs)
            
            next_obs, rewards, done, info = env.step([action1, action2])
            
            # Store experiences
            agent1.store_experience(obs, action1, rewards[0], next_obs, done, log_prob1, value1)
            agent2.store_experience(obs, action2, rewards[1], next_obs, done, log_prob2, value2)
            
            obs = next_obs
            
            if done:
                break
        
        print("✓ Quick training test completed")
        return True
    except Exception as e:
        print(f"✗ Quick training test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("MAPPO Implementation Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_device_detection,
        test_mappo_agent,
        test_environment,
        test_logger,
        test_parallel_manager,
        run_quick_training_test
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed! MAPPO implementation is ready.")
        print("\nNext steps:")
        print("1. Run demo: python main.py --mode demo")
        print("2. Start training: python main.py --mode train --episodes 100")
        print("3. Evaluate: python main.py --mode eval --render")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
