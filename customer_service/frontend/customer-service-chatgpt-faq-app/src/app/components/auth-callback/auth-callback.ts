import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';

@Component({
  selector: 'app-auth-callback',
  template: `
    <div class="auth-callback-container">
      <div class="loading-spinner" *ngIf="isProcessing">
        <div class="spinner"></div>
        <p>Processing authentication...</p>
      </div>
      
      <div class="error-message" *ngIf="errorMessage">
        <h3>Authentication Error</h3>
        <p>{{ errorMessage }}</p>
        <button (click)="redirectToLogin()">Try Again</button>
      </div>

      <div class="success-message" *ngIf="successMessage">
        <h3>Authentication Successful</h3>
        <p>{{ successMessage }}</p>
        <p>Redirecting to main application...</p>
      </div>
    </div>
  `,
  styles: [`
    .auth-callback-container {
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      padding: 20px;
      text-align: center;
    }

    .loading-spinner {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 20px;
    }

    .spinner {
      width: 40px;
      height: 40px;
      border: 4px solid #f3f3f3;
      border-top: 4px solid #3498db;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    .error-message, .success-message {
      background: white;
      padding: 30px;
      border-radius: 8px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      max-width: 400px;
    }

    .error-message {
      border-left: 4px solid #e74c3c;
    }

    .success-message {
      border-left: 4px solid #27ae60;
    }

    button {
      background-color: #3498db;
      color: white;
      border: none;
      padding: 10px 20px;
      border-radius: 4px;
      cursor: pointer;
      margin-top: 15px;
    }

    button:hover {
      background-color: #2980b9;
    }
  `],
  standalone: true,
  imports: [CommonModule]
})
export class AuthCallbackComponent implements OnInit {
  isProcessing = true;
  errorMessage = '';
  successMessage = '';

  constructor(
    private route: ActivatedRoute,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.handleAuthCallback();
  }

  private async handleAuthCallback(): Promise<void> {
    try {
      // Get query parameters from URL
      this.route.queryParams.subscribe(async params => {
        const token = params['token'];
        const expiresIn = params['expires_in'];
        const error = params['error'];
        const errorDescription = params['error_description'];

        if (error) {
          this.handleAuthError(error, errorDescription);
          return;
        }

        if (!token) {
          this.handleAuthError('missing_token', 'No authentication token received');
          return;
        }

        // Validate token with backend
        const isValid = await this.validateToken(token);
        
        if (isValid) {
          // Store token and expiration
          this.storeAuthToken(token, expiresIn);
          this.handleAuthSuccess();
        } else {
          this.handleAuthError('invalid_token', 'Token validation failed');
        }
      });
    } catch (error) {
      this.handleAuthError('processing_error', 'Failed to process authentication');
    }
  }

  private async validateToken(token: string): Promise<boolean> {
    try {
      const response = await fetch('http://127.0.0.1:8000/auth/validate-token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      });

      return response.ok;
    } catch (error) {
      console.error('Token validation failed:', error);
      return false;
    }
  }

  private storeAuthToken(token: string, expiresIn?: string): void {
    // Store in localStorage for persistence
    localStorage.setItem('auth_token', token);
    
    // Calculate and store expiration time
    if (expiresIn) {
      const expirationTime = Date.now() + (parseInt(expiresIn) * 1000);
      localStorage.setItem('auth_token_expires', expirationTime.toString());
    }

    // Also store in sessionStorage as fallback
    sessionStorage.setItem('auth_token', token);

    // Clear any previous auth URL
    sessionStorage.removeItem('auth-url');
  }

  private handleAuthSuccess(): void {
    this.isProcessing = false;
    this.successMessage = 'Authentication completed successfully!';
    
    // Redirect to main app after 2 seconds
    setTimeout(() => {
      this.router.navigate(['/']);
    }, 2000);
  }

  private handleAuthError(error: string, description?: string): void {
    this.isProcessing = false;
    this.errorMessage = description || `Authentication failed: ${error}`;
  }

  redirectToLogin(): void {
    // Get the original auth URL from session storage if available
    const authUrl = sessionStorage.getItem('auth-url');
    if (authUrl) {
      window.location.href = authUrl;
    } else {
      // Fallback to main app
      this.router.navigate(['/']);
    }
  }
}