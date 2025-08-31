import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, Subject } from 'rxjs';

export interface MCPToolCall {
  name: string;
  arguments: any;
}

export interface MCPResponse {
  jsonrpc: string;
  id: string | number;
  result?: any;
  error?: {
    code: number;
    message: string;
  };
}

export interface MCPConnectionState {
  connected: boolean;
  authenticated: boolean;
  error?: string;
}

@Injectable({
  providedIn: 'root'
})
export class McpClientService {
  private ws: WebSocket | null = null;
  private messageId = 0;
  private pendingRequests = new Map<string | number, {
    resolve: (value: any) => void;
    reject: (error: any) => void;
  }>();

  // Observables for connection state
  private connectionStateSubject = new BehaviorSubject<MCPConnectionState>({
    connected: false,
    authenticated: false
  });
  
  private messageSubject = new Subject<MCPResponse>();
  
  public connectionState$ = this.connectionStateSubject.asObservable();
  public messages$ = this.messageSubject.asObservable();

  constructor() {}

  /**
   * Connect to MCP WebSocket server using token-based authentication
   */
  async connectWithToken(token: string): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        console.log('Attempting to connect to MCP WebSocket with token...');
        
        // Connect to WebSocket with token in query parameter
        this.ws = new WebSocket(`ws://localhost:8000/mcp-ws?token=${token}`);
        
        // Set up timeout for connection
        const connectionTimeout = setTimeout(() => {
          console.error('MCP WebSocket connection timeout');
          this.ws?.close();
          reject(new Error('Connection timeout'));
        }, 10000); // 10 second timeout
        
        this.ws.onopen = () => {
          console.log('=== MCP WEBSOCKET OPENED ===');
          console.log('WebSocket readyState after open:', this.ws?.readyState);
          clearTimeout(connectionTimeout);
          
          const newState = {
            connected: true,
            authenticated: true
          };
          
          console.log('Updating connection state to:', newState);
          this.updateConnectionState(newState);
          console.log('Connection state updated, resolving promise...');
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const response: MCPResponse = JSON.parse(event.data);
            console.log('MCP Response received:', response);
            
            // Check for authentication errors
            if (response.error && response.error.code === 4001) {
              console.error('MCP Authentication failed:', response.error.message);
              this.updateConnectionState({
                connected: false,
                authenticated: false,
                error: 'Authentication failed: ' + response.error.message
              });
              return;
            }
            
            // Emit to message stream
            this.messageSubject.next(response);
            
            // Resolve pending request if this is a response to a request
            if (response.id && this.pendingRequests.has(response.id)) {
              const pending = this.pendingRequests.get(response.id)!;
              this.pendingRequests.delete(response.id);
              
              if (response.error) {
                pending.reject(new Error(response.error.message));
              } else {
                pending.resolve(response.result);
              }
            }
          } catch (error) {
            console.error('Error parsing MCP message:', error);
          }
        };

        this.ws.onclose = (event) => {
          console.log('MCP WebSocket closed:', event.code, event.reason);
          clearTimeout(connectionTimeout);
          
          let errorMessage = 'Connection closed';
          if (event.code === 4001) {
            errorMessage = 'Authentication failed';
          } else if (event.reason) {
            errorMessage = event.reason;
          } else if (event.code !== 1000) {
            errorMessage = `Connection closed unexpectedly (${event.code})`;
          }
          
          this.updateConnectionState({
            connected: false,
            authenticated: false,
            error: errorMessage
          });
          this.ws = null;
          
          if (event.code !== 1000) {
            reject(new Error(errorMessage));
          }
        };

        this.ws.onerror = (error) => {
          console.error('MCP WebSocket error:', error);
          clearTimeout(connectionTimeout);
          this.updateConnectionState({
            connected: false,
            authenticated: false,
            error: 'WebSocket connection failed'
          });
          reject(new Error('WebSocket connection failed'));
        };

      } catch (error) {
        console.error('Error creating WebSocket:', error);
        reject(error);
      }
    });
  }

  /**
   * Connect to MCP WebSocket server without authentication (for public tools)
   */
  async connectWithoutAuth(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        console.log('Attempting to connect to MCP WebSocket without authentication...');
        
        // Connect to WebSocket without token (for public tools)
        this.ws = new WebSocket('ws://localhost:8000/mcp-ws');
        
        // Set up timeout for connection
        const connectionTimeout = setTimeout(() => {
          console.error('MCP WebSocket connection timeout');
          this.ws?.close();
          reject(new Error('Connection timeout'));
        }, 10000); // 10 second timeout
        
        this.ws.onopen = () => {
          console.log('=== MCP WEBSOCKET OPENED (UNAUTHENTICATED) ===');
          console.log('WebSocket readyState after open:', this.ws?.readyState);
          clearTimeout(connectionTimeout);
          
          const newState = {
            connected: true,
            authenticated: false // Not authenticated, only public tools
          };
          
          console.log('Updating connection state to:', newState);
          this.updateConnectionState(newState);
          console.log('Connection state updated, resolving promise...');
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const response: MCPResponse = JSON.parse(event.data);
            console.log('MCP Response received:', response);
            
            // Check for authentication errors (401 for protected tools)
            if (response.error && response.error.code === 401) {
              console.log('Tool requires authentication:', response.error.message);
              // Don't close connection, just handle the auth error for this request
            }
            
            // Emit to message stream
            this.messageSubject.next(response);
            
            // Resolve pending request if this is a response to a request
            if (response.id && this.pendingRequests.has(response.id)) {
              const pending = this.pendingRequests.get(response.id)!;
              this.pendingRequests.delete(response.id);
              
              if (response.error) {
                pending.reject(new Error(response.error.message));
              } else {
                pending.resolve(response.result);
              }
            }
          } catch (error) {
            console.error('Error parsing MCP message:', error);
          }
        };

        this.ws.onclose = (event) => {
          console.log('MCP WebSocket closed:', event.code, event.reason);
          clearTimeout(connectionTimeout);
          
          let errorMessage = 'Connection closed';
          if (event.reason) {
            errorMessage = event.reason;
          } else if (event.code !== 1000) {
            errorMessage = `Connection closed unexpectedly (${event.code})`;
          }
          
          this.updateConnectionState({
            connected: false,
            authenticated: false,
            error: errorMessage
          });
          this.ws = null;
          
          if (event.code !== 1000) {
            reject(new Error(errorMessage));
          }
        };

        this.ws.onerror = (error) => {
          console.error('MCP WebSocket error:', error);
          clearTimeout(connectionTimeout);
          this.updateConnectionState({
            connected: false,
            authenticated: false,
            error: 'WebSocket connection failed'
          });
          reject(new Error('WebSocket connection failed'));
        };

      } catch (error) {
        console.error('Error creating WebSocket:', error);
        reject(error);
      }
    });
  }

  /**
   * Connect to MCP WebSocket server (deprecated - use connectWithToken or connectWithoutAuth)
   */
  async connect(): Promise<void> {
    throw new Error('Use connectWithToken() or connectWithoutAuth() instead of connect()');
  }

  /**
   * Disconnect from MCP server
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    this.updateConnectionState({
      connected: false,
      authenticated: false
    });
    
    // Reject all pending requests
    this.pendingRequests.forEach(({ reject }) => {
      reject(new Error('Connection closed'));
    });
    this.pendingRequests.clear();
  }

  /**
   * Initialize MCP session
   */
  async initialize(): Promise<any> {
    const request = {
      jsonrpc: "2.0",
      method: "initialize",
      params: {
        protocolVersion: "2024-11-05",
        capabilities: {},
        clientInfo: {
          name: "customer-service-angular-app",
          version: "1.0.0"
        }
      },
      id: ++this.messageId
    };

    return this.sendRequest(request);
  }

  /**
   * List available tools
   */
  async listTools(): Promise<any> {
    const request = {
      jsonrpc: "2.0",
      method: "tools/list",
      params: {},
      id: ++this.messageId
    };

    return this.sendRequest(request);
  }

  /**
   * Call a specific tool
   */
  async callTool(toolCall: MCPToolCall): Promise<any> {
    const request = {
      jsonrpc: "2.0",
      method: "tools/call",
      params: {
        name: toolCall.name,
        arguments: toolCall.arguments
      },
      id: ++this.messageId
    };

    return this.sendRequest(request);
  }

  /**
   * Send a request and return a promise that resolves with the response
   */
  private sendRequest(request: any): Promise<any> {
    return new Promise((resolve, reject) => {
      console.log('=== SENDING MCP REQUEST ===');
      console.log('WebSocket exists:', !!this.ws);
      console.log('WebSocket readyState:', this.ws?.readyState);
      console.log('WebSocket OPEN constant:', WebSocket.OPEN);
      console.log('Connection state:', this.getConnectionState());
      
      if (!this.ws) {
        const error = 'WebSocket object does not exist';
        console.error('ERROR:', error);
        reject(new Error(error));
        return;
      }
      
      if (this.ws.readyState !== WebSocket.OPEN) {
        const error = `WebSocket not in OPEN state. Current state: ${this.ws.readyState}`;
        console.error('ERROR:', error);
        console.log('WebSocket states: CONNECTING=0, OPEN=1, CLOSING=2, CLOSED=3');
        reject(new Error(error));
        return;
      }

      console.log('WebSocket is ready, sending request...');

      // Store the promise resolvers
      this.pendingRequests.set(request.id, { resolve, reject });

      // Send the request
      try {
        this.ws.send(JSON.stringify(request));
        console.log('MCP Request sent successfully:', request);
      } catch (sendError) {
        console.error('Error sending WebSocket message:', sendError);
        this.pendingRequests.delete(request.id);
        reject(sendError);
        return;
      }

      // Set timeout for request
      setTimeout(() => {
        if (this.pendingRequests.has(request.id)) {
          this.pendingRequests.delete(request.id);
          console.error('Request timeout for request ID:', request.id);
          reject(new Error('Request timeout'));
        }
      }, 30000); // 30 second timeout
    });
  }

  /**
   * Update connection state and notify subscribers
   */
  private updateConnectionState(state: MCPConnectionState): void {
    this.connectionStateSubject.next(state);
  }

  /**
   * Check if currently connected (regardless of authentication state)
   */
  isConnected(): boolean {
    const state = this.connectionStateSubject.value;
    return state.connected; // Only check if connected, not authentication state
  }

  /**
   * Check if currently connected and authenticated
   */
  isAuthenticated(): boolean {
    const state = this.connectionStateSubject.value;
    return state.connected && state.authenticated;
  }

  /**
   * Get current connection state
   */
  getConnectionState(): MCPConnectionState {
    return this.connectionStateSubject.value;
  }
}