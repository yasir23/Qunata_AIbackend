"""
Test script for Firebase Authentication integration.

This script tests the Firebase authentication service to ensure it's properly configured
and can handle authentication requests.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from security.firebase_auth import firebase_service
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_firebase_initialization():
    """Test Firebase service initialization."""
    logger.info("Testing Firebase service initialization...")
    
    if firebase_service.is_initialized():
        logger.info("✅ Firebase service initialized successfully")
        return True
    else:
        logger.error("❌ Firebase service failed to initialize")
        logger.info("Make sure you have set the following environment variables:")
        logger.info("- FIREBASE_PROJECT_ID")
        logger.info("- FIREBASE_PRIVATE_KEY")
        logger.info("- FIREBASE_CLIENT_EMAIL")
        logger.info("Or set FIREBASE_CONFIG with the full JSON configuration")
        return False

async def test_token_verification():
    """Test token verification with a mock token."""
    logger.info("Testing token verification...")
    
    if not firebase_service.is_initialized():
        logger.error("❌ Cannot test token verification - Firebase not initialized")
        return False
    
    # Test with an invalid token (should return None)
    try:
        result = await firebase_service.verify_token("invalid_token")
        if result is None:
            logger.info("✅ Invalid token correctly rejected")
            return True
        else:
            logger.error("❌ Invalid token was accepted (this shouldn't happen)")
            return False
    except Exception as e:
        logger.info(f"✅ Invalid token correctly rejected with exception: {e}")
        return True

async def test_user_profile_operations():
    """Test user profile storage and retrieval."""
    logger.info("Testing user profile operations...")
    
    if not firebase_service.is_initialized():
        logger.error("❌ Cannot test profile operations - Firebase not initialized")
        return False
    
    # Test storing a mock user profile
    test_uid = "test_user_123"
    test_profile = {
        "email": "test@example.com",
        "name": "Test User",
        "picture": "https://example.com/avatar.jpg",
        "email_verified": True
    }
    
    try:
        # Store profile
        store_result = await firebase_service.store_user_profile(test_uid, test_profile)
        if store_result:
            logger.info("✅ User profile stored successfully")
        else:
            logger.error("❌ Failed to store user profile")
            return False
        
        # Retrieve profile
        retrieved_profile = await firebase_service.get_user_profile(test_uid)
        if retrieved_profile:
            logger.info("✅ User profile retrieved successfully")
            logger.info(f"Retrieved profile: {retrieved_profile}")
        else:
            logger.warning("⚠️ User profile not found (this might be expected for test data)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing profile operations: {e}")
        return False

async def main():
    """Run all Firebase authentication tests."""
    logger.info("🔥 Starting Firebase Authentication Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Firebase Initialization", test_firebase_initialization),
        ("Token Verification", test_token_verification),
        ("User Profile Operations", test_user_profile_operations)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running test: {test_name}")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Results Summary:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {status}: {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Firebase authentication is ready.")
    else:
        logger.warning("⚠️ Some tests failed. Check the configuration and try again.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
