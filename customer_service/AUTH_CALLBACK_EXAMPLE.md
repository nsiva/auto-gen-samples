# Authentication Callback Implementation

## Overview
This implementation provides a complete authentication callback flow using the "Redirect with Token" approach for the customer service application.

## How It Works

### 1. Authentication Required Response
When the backend detects that authentication is required (e.g., for protected endpoints), it returns:
- Status: `401 Unauthorized`
- Header: `X-Auth-URL: https://your-auth-provider.com/oauth/authorize?client_id=...&redirect_uri=http://localhost:4200/auth/callback`

### 2. User Authentication Flow
1. User clicks on authentication link/button
2. Redirected to external auth provider
3. After successful authentication, auth provider redirects to: `http://localhost:4200/auth/callback?token=abc123&expires_in=3600`

### 3. Callback Processing
The Angular callback component (`AuthCallbackComponent`) handles the redirect:
- Extracts token and expiration from URL parameters
- Validates token with backend endpoint `/auth/validate-token`
- Stores token in localStorage and sessionStorage
- Redirects user back to main application

### 4. Subsequent API Calls
All future API calls automatically include the token in the Authorization header:
```javascript
headers: {
  'Authorization': 'Bearer abc123'
}
```

## Components Created

### Frontend (`Angular`)
- **AuthCallbackComponent**: Handles auth redirect processing
- **App Router**: Routes `/auth/callback` to callback component
- **Token Management**: Storage, validation, and automatic inclusion in API calls

### Backend (`FastAPI`)
- **Token Validation Endpoint**: `/auth/validate-token` - validates tokens received from auth provider
- **Protected Endpoints**: Existing endpoints that require authentication (e.g., `/ask`)

## Usage Example

### 1. User makes a query requiring authentication
```javascript
// This will trigger 401 if authentication is required
const response = await fetch('/ask', {
  method: 'POST',
  body: JSON.stringify({query: "What is the status of order 123?"})
});

if (response.status === 401) {
  const authUrl = response.headers.get('X-Auth-URL');
  // Redirect to auth provider
  window.location.href = authUrl;
}
```

### 2. Authentication provider redirects back
```
http://localhost:4200/auth/callback?token=eyJ0eXAiOiJKV1QiLCJhbGc&expires_in=3600
```

### 3. Callback component processes the token
- Validates with backend
- Stores token locally
- Redirects to main app

### 4. Subsequent requests are automatically authenticated
```javascript
// Token automatically included in headers
const response = await fetch('/ask', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGc...'
  },
  body: JSON.stringify({query: "What is the status of order 123?"})
});
```

## Security Features

- **Token Expiration**: Tokens are automatically removed when expired
- **Backend Validation**: All tokens are validated server-side
- **Dual Storage**: Tokens stored in both localStorage (persistence) and sessionStorage (backup)
- **Error Handling**: Comprehensive error handling for invalid tokens, network errors, etc.

## Testing the Implementation

1. Start the backend server: `cd backend && ./start_server`
2. Start the frontend: `cd frontend/customer-service-chatgpt-faq-app && npm start`
3. Navigate to `http://localhost:4200`
4. Ask a question that requires authentication (e.g., "What's the status of order 123?")
5. The system will guide you through the authentication flow

## URL Pattern
The callback URL follows the standard OAuth pattern:
```
http://localhost:4200/auth/callback?token={JWT_TOKEN}&expires_in={SECONDS}
```

Additional parameters supported:
- `error`: Error code if authentication failed
- `error_description`: Human-readable error description