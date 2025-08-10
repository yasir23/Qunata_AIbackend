"""
Firebase Authentication module for Open Deep Research.

This module provides Firebase Authentication integration with Google OAuth support,
JWT token validation, user management, and session handling.
"""

import os
import json
import asyncio
from typing import Optional, Dict, Any
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
from google.cloud import firestore
import logging

logger = logging.getLogger(__name__)

class FirebaseAuthService:
    """Firebase Authentication service for user management and token validation."""
    
    def __init__(self):
        self.app = None
        self.db = None
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK."""
        try:
            # Check if Firebase is already initialized
            if firebase_admin._apps:
                self.app = firebase_admin.get_app()
            else:
                # Initialize from environment variables
                firebase_config = self._get_firebase_config()
                if firebase_config:
                    cred = credentials.Certificate(firebase_config)
                    self.app = firebase_admin.initialize_app(cred)
                else:
                    # Try to initialize with default credentials (for local development)
                    self.app = firebase_admin.initialize_app()
            
            # Initialize Firestore client
            self.db = firestore.Client()
            logger.info("Firebase initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            self.app = None
            self.db = None
    
    def _get_firebase_config(self) -> Optional[Dict[str, Any]]:
        """Get Firebase configuration from environment variables."""
        try:
            # Try to get config from environment variable (JSON string)
            firebase_config_json = os.environ.get("FIREBASE_CONFIG")
            if firebase_config_json:
                return json.loads(firebase_config_json)
            
            # Try to get individual config values
            project_id = os.environ.get("FIREBASE_PROJECT_ID")
            private_key = os.environ.get("FIREBASE_PRIVATE_KEY")
            client_email = os.environ.get("FIREBASE_CLIENT_EMAIL")
            
            if project_id and private_key and client_email:
                # Replace escaped newlines in private key
                private_key = private_key.replace('\\n', '\n')
                
                return {
                    "type": "service_account",
                    "project_id": project_id,
                    "private_key": private_key,
                    "client_email": client_email,
                    "client_id": os.environ.get("FIREBASE_CLIENT_ID", ""),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{client_email}"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing Firebase config: {e}")
            return None
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify Firebase ID token and return user information.
        
        Args:
            token: Firebase ID token
            
        Returns:
            User information dict or None if invalid
        """
        if not self.app:
            raise Exception("Firebase not initialized")
        
        try:
            # Verify the token in a separate thread to avoid blocking
            decoded_token = await asyncio.to_thread(
                firebase_auth.verify_id_token, token
            )
            
            # Get additional user info
            user_record = await asyncio.to_thread(
                firebase_auth.get_user, decoded_token['uid']
            )
            
            return {
                'uid': decoded_token['uid'],
                'email': decoded_token.get('email'),
                'email_verified': decoded_token.get('email_verified', False),
                'name': decoded_token.get('name'),
                'picture': decoded_token.get('picture'),
                'provider_data': [
                    {
                        'provider_id': provider.provider_id,
                        'uid': provider.uid,
                        'email': provider.email,
                        'display_name': provider.display_name,
                        'photo_url': provider.photo_url
                    }
                    for provider in user_record.provider_data
                ],
                'custom_claims': decoded_token.get('custom_claims', {}),
                'auth_time': decoded_token.get('auth_time'),
                'iat': decoded_token.get('iat'),
                'exp': decoded_token.get('exp')
            }
            
        except firebase_auth.InvalidIdTokenError:
            logger.warning("Invalid Firebase ID token")
            return None
        except firebase_auth.ExpiredIdTokenError:
            logger.warning("Expired Firebase ID token")
            return None
        except Exception as e:
            logger.error(f"Error verifying Firebase token: {e}")
            return None
    
    async def get_user_by_uid(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        Get user information by UID.
        
        Args:
            uid: Firebase user UID
            
        Returns:
            User information dict or None if not found
        """
        if not self.app:
            raise Exception("Firebase not initialized")
        
        try:
            user_record = await asyncio.to_thread(
                firebase_auth.get_user, uid
            )
            
            return {
                'uid': user_record.uid,
                'email': user_record.email,
                'email_verified': user_record.email_verified,
                'display_name': user_record.display_name,
                'photo_url': user_record.photo_url,
                'disabled': user_record.disabled,
                'metadata': {
                    'creation_timestamp': user_record.user_metadata.creation_timestamp,
                    'last_sign_in_timestamp': user_record.user_metadata.last_sign_in_timestamp,
                    'last_refresh_timestamp': user_record.user_metadata.last_refresh_timestamp
                },
                'provider_data': [
                    {
                        'provider_id': provider.provider_id,
                        'uid': provider.uid,
                        'email': provider.email,
                        'display_name': provider.display_name,
                        'photo_url': provider.photo_url
                    }
                    for provider in user_record.provider_data
                ],
                'custom_claims': user_record.custom_claims or {}
            }
            
        except firebase_auth.UserNotFoundError:
            logger.warning(f"User not found: {uid}")
            return None
        except Exception as e:
            logger.error(f"Error getting user by UID: {e}")
            return None
    
    async def create_custom_token(self, uid: str, additional_claims: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Create a custom token for a user.
        
        Args:
            uid: Firebase user UID
            additional_claims: Additional claims to include in the token
            
        Returns:
            Custom token string or None if failed
        """
        if not self.app:
            raise Exception("Firebase not initialized")
        
        try:
            custom_token = await asyncio.to_thread(
                firebase_auth.create_custom_token, uid, additional_claims
            )
            return custom_token.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error creating custom token: {e}")
            return None
    
    async def set_custom_user_claims(self, uid: str, custom_claims: Dict[str, Any]) -> bool:
        """
        Set custom claims for a user.
        
        Args:
            uid: Firebase user UID
            custom_claims: Custom claims to set
            
        Returns:
            True if successful, False otherwise
        """
        if not self.app:
            raise Exception("Firebase not initialized")
        
        try:
            await asyncio.to_thread(
                firebase_auth.set_custom_user_claims, uid, custom_claims
            )
            return True
            
        except Exception as e:
            logger.error(f"Error setting custom user claims: {e}")
            return False
    
    async def store_user_profile(self, uid: str, profile_data: Dict[str, Any]) -> bool:
        """
        Store user profile data in Firestore.
        
        Args:
            uid: Firebase user UID
            profile_data: Profile data to store
            
        Returns:
            True if successful, False otherwise
        """
        if not self.db:
            logger.error("Firestore not initialized")
            return False
        
        try:
            # Add timestamp
            profile_data['updated_at'] = firestore.SERVER_TIMESTAMP
            if 'created_at' not in profile_data:
                profile_data['created_at'] = firestore.SERVER_TIMESTAMP
            
            # Store in Firestore
            await asyncio.to_thread(
                self.db.collection('users').document(uid).set,
                profile_data,
                merge=True
            )
            return True
            
        except Exception as e:
            logger.error(f"Error storing user profile: {e}")
            return False
    
    async def get_user_profile(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        Get user profile data from Firestore.
        
        Args:
            uid: Firebase user UID
            
        Returns:
            Profile data dict or None if not found
        """
        if not self.db:
            logger.error("Firestore not initialized")
            return None
        
        try:
            doc_ref = self.db.collection('users').document(uid)
            doc = await asyncio.to_thread(doc_ref.get)
            
            if doc.exists:
                return doc.to_dict()
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None
    
    def is_initialized(self) -> bool:
        """Check if Firebase is properly initialized."""
        return self.app is not None and self.db is not None


# Global Firebase service instance
firebase_service = FirebaseAuthService()
