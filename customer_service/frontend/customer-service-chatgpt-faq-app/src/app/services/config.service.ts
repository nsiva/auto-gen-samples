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
   * or falls back to default URL
   */
  getAuthUrl(): string {
    // Try to get auth URL from session storage (set during 401 responses)
    const storedAuthUrl = sessionStorage.getItem('auth-url');
    if (storedAuthUrl) {
      return storedAuthUrl;
    }
    
    // Fallback to default URL
    return this.DEFAULT_AUTH_URL;
  }
}