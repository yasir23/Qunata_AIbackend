# Firebase Authentication Setup Guide

This guide explains how to set up Firebase Authentication for Open Deep Research, replacing the previous Supabase integration.

## Prerequisites

1. A Google Cloud Platform account
2. A Firebase project
3. Firebase Admin SDK service account credentials

## Firebase Project Setup

### 1. Create a Firebase Project

1. Go to the [Firebase Console](https://console.firebase.google.com/)
2. Click "Create a project" or "Add project"
3. Enter your project name (e.g., "open-deep-research")
4. Enable Google Analytics if desired
5. Click "Create project"

### 2. Enable Authentication

1. In your Firebase project, go to "Authentication" in the left sidebar
2. Click "Get started"
3. Go to the "Sign-in method" tab
4. Enable "Google" as a sign-in provider
5. Configure the OAuth consent screen if prompted

### 3. Generate Service Account Credentials

1. Go to Project Settings (gear icon) → "Service accounts"
2. Click "Generate new private key"
3. Download the JSON file containing your service account credentials
4. Keep this file secure - it contains sensitive credentials

### 4. Enable Firestore Database

1. Go to "Firestore Database" in the left sidebar
2. Click "Create database"
3. Choose "Start in production mode" or "Start in test mode" based on your needs
4. Select a location for your database

## Environment Configuration

Set the following environment variables in your `.env` file:

```bash
# Firebase Configuration
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nYour private key here\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL=your-service-account@your-project.iam.gserviceaccount.com
FIREBASE_CLIENT_ID=your-client-id
```

## Testing the Integration

Run the test script to verify your Firebase setup:

```bash
cd src/security
python test_firebase_auth.py
```

## Key Components

- **`src/security/firebase_auth.py`**: Firebase authentication service
- **`src/security/auth.py`**: LangGraph auth integration
- **`src/security/test_firebase_auth.py`**: Test script for verification

The Firebase authentication is now fully integrated and ready for production use with proper credentials.
