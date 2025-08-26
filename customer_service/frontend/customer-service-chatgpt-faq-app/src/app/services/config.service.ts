import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ConfigService {
  private readonly API_BASE_URL = 'http://127.0.0.1:8000';

  // Default fallback auth URL (will be replaced by backend config)
  public readonly DEFAULT_AUTH_URL = 'http://localhost:4201/login';

  constructor() { }

  /**
   * Get the authentication URL from session storage (set during 401 responses) 
   * or falls back to default URL, with callback URL as query parameter
   */
  getAuthUrl(): string {
    // Define the callback URL
    const callbackUrl = `${window.location.origin}/auth/callback`;
    
    // Try to get auth URL from session storage (set during 401 responses)
    const storedAuthUrl = sessionStorage.getItem('auth-url');
    let baseAuthUrl = storedAuthUrl || this.DEFAULT_AUTH_URL;
    
    // Add callback URL as query parameter
    const separator = baseAuthUrl.includes('?') ? '&' : '?';
    return `${baseAuthUrl}${separator}callback_url=${encodeURIComponent(callbackUrl)}`;
  }
}