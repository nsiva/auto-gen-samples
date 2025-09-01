import { Component, Input, Output, EventEmitter, ViewChild, ElementRef, AfterViewChecked, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { StreamingQueryService } from '../../services/streaming-query.service';
import { ConfigService } from '../../services/config.service';
import { McpClientService, MCPConnectionState } from '../../services/mcp-client.service';
import { StreamEvent, StreamingResponse, ToolProgress } from '../../models/streaming.models';
import { Subscription } from 'rxjs';

export interface ChatMessage {
  text: string;
  sender: 'user' | 'bot';
  streaming?: boolean;
  progress?: ToolProgress;
  streamEvents?: StreamEvent[];
  actionEvents?: ActionEvent[];
  error?: boolean;
  requiresAuth?: boolean;
}

export interface ActionEvent {
  type: string;
  timestamp: string;
  data: any;
}

@Component({
  selector: 'app-chat-interface',
  templateUrl: './chat-interface.html',
  styleUrls: ['./chat-interface.scss'],
  standalone: true,
  imports: [CommonModule, FormsModule]
})
export class ChatInterface implements AfterViewChecked, OnInit, OnDestroy {
  @Input() messages: ChatMessage[] = [];
  @Input() queryText: string = ''; // Input for pre-filling from FAQ
  @Output() querySubmit = new EventEmitter<string>();
  @Output() queryTextChange = new EventEmitter<string>(); // For two-way binding or direct update

  @ViewChild('messageContainer') private messageContainer!: ElementRef;

  // Streaming properties
  public isStreaming = false;
  public streamingProgress: ToolProgress = { step: 0, total: 0, current: '', percentage: 0 };
  public currentStreamingMessage: ChatMessage | null = null;

  // MCP connection properties
  public isMcpProcessing = false;
  public mcpConnectionState: MCPConnectionState = {
    connected: false,
    authenticated: false
  };
  private mcpSubscription: Subscription | null = null;

  // Command history properties
  private readonly HISTORY_KEY = 'chat_input_history';
  private inputHistory: string[] = [];
  private historyIndex: number = -1;

  constructor(
    private streamingQueryService: StreamingQueryService,
    private configService: ConfigService,
    private mcpClient: McpClientService
  ) {
    this.loadInputHistory();
  }

  ngOnInit() {
    // Subscribe to MCP connection state
    this.mcpSubscription = this.mcpClient.connectionState$.subscribe(
      state => {
        this.mcpConnectionState = state;
        console.log('MCP Connection State:', state);
      }
    );
    
    // Try to initialize MCP connection (authenticated if possible, otherwise public)
    this.connectToMcp().catch(error => {
      console.log('MCP initialization failed on page load:', error);
      // Don't show error messages on page load, just update state
    });
  }

  ngOnDestroy() {
    if (this.mcpSubscription) {
      this.mcpSubscription.unsubscribe();
    }
    this.mcpClient.disconnect();
  }

  ngAfterViewChecked() {
    this.scrollToBottom();
  }

  onSendClick(): void {
    if (this.queryText.trim()) {
      this.addToHistory(this.queryText.trim());
      this.querySubmit.emit(this.queryText.trim());
      this.queryText = ''; // Clear input after sending
      this.queryTextChange.emit(''); // Notify parent about cleared text
      this.resetHistoryIndex();
    }
  }

  async onSendStreamingClick(): Promise<void> {
    if (this.queryText.trim() && !this.isStreaming) {
      const userQuery = this.queryText.trim();
      
      // Add to history before processing
      this.addToHistory(userQuery);
      
      // Add user message to chat
      const userMessage: ChatMessage = {
        text: userQuery,
        sender: 'user'
      };
      this.messages.push(userMessage);
      
      // Clear input
      this.queryText = '';
      this.queryTextChange.emit('');
      this.resetHistoryIndex();
      
      // Initialize streaming bot message
      this.currentStreamingMessage = {
        text: 'Processing your request...',
        sender: 'bot',
        streaming: true,
        progress: { step: 0, total: 0, current: 'Starting...', percentage: 0 },
        streamEvents: []
      };
      this.messages.push(this.currentStreamingMessage);
      
      this.isStreaming = true;
      
      try {
        const streamObservable = await this.streamingQueryService.submitStreamingQuery(
          userQuery,
          this.getConversationHistory()
        );
        
        streamObservable.subscribe({
          next: (response: StreamingResponse) => {
            this.handleStreamingResponse(response);
          },
          error: (error) => {
            console.error('Streaming error:', error);
            this.handleStreamingError(error);
          },
          complete: () => {
            this.handleStreamingComplete();
          }
        });
        
      } catch (error) {
        console.error('Failed to start streaming:', error);
        this.handleStreamingError(error);
      }
    }
  }

  onInputChange(event: Event): void {
    this.queryText = (event.target as HTMLInputElement).value;
    this.queryTextChange.emit(this.queryText); // Emit changes to parent
  }

  onEnterPress(event: Event): void {
    const keyboardEvent = event as KeyboardEvent;
    if (keyboardEvent.key === 'Enter' && !keyboardEvent.shiftKey) {
      keyboardEvent.preventDefault();
      this.onSendClick();
    }
  }

  onArrowKeyPress(event: Event): void {
    const keyboardEvent = event as KeyboardEvent;
    if (keyboardEvent.key === 'ArrowUp') {
      keyboardEvent.preventDefault();
      this.navigateHistory('up');
    } else if (keyboardEvent.key === 'ArrowDown') {
      keyboardEvent.preventDefault();
      this.navigateHistory('down');
    }
  }

  async onSendMcpClick(): Promise<void> {
    console.log('=== MCP SEND BUTTON CLICKED ===');
    
    if (this.queryText.trim() && !this.isMcpProcessing) {
      const userQuery = this.queryText.trim();
      console.log('Query to send:', userQuery);
      
      // Check if MCP is connected
      const isConnected = this.mcpClient.isConnected();
      console.log('MCP connected status:', isConnected);
      console.log('Current connection state:', this.mcpConnectionState);
      
      if (!isConnected) {
        console.log('MCP not connected, attempting to connect...');
        // Try to connect (with or without auth)
        try {
          await this.connectToMcp();
          console.log('Connection successful');
        } catch (error) {
          console.error('Connection failed:', error);
          this.messages.push({
            text: `Failed to connect to MCP: ${error}`,
            sender: 'bot',
            error: true
          });
          return;
        }
      } else {
        console.log('MCP already connected, proceeding with query...');
      }
      
      // Add to history
      this.addToHistory(userQuery);
      
      // Add user message to chat
      const userMessage: ChatMessage = {
        text: userQuery,
        sender: 'user'
      };
      this.messages.push(userMessage);
      
      // Clear input
      this.queryText = '';
      this.queryTextChange.emit('');
      this.resetHistoryIndex();
      
      // Process with MCP
      console.log('About to process MCP query...');
      await this.processMcpQuery(userQuery);
    } else {
      console.log('Cannot send MCP query:', {
        hasQuery: !!this.queryText.trim(),
        isProcessing: this.isMcpProcessing
      });
    }
  }

  private async connectToMcp(): Promise<void> {
    console.log('=== CONNECTING TO MCP ===');
    
    const authToken = localStorage.getItem('auth_token');
    console.log('Auth token available:', !!authToken);
    
    if (authToken) {
      // Authenticated connection
      console.log('Attempting authenticated MCP connection...');
      await this.initializeMcpConnectionWithAuth();
    } else {
      // Unauthenticated connection for public tools
      console.log('Attempting unauthenticated MCP connection for public tools...');
      await this.initializeMcpConnectionWithoutAuth();
    }
  }

  private async initializeMcpConnectionWithAuth(): Promise<void> {
    console.log('=== INITIALIZING AUTHENTICATED MCP CONNECTION ===');
    
    try {
      // Get the MCP token
      console.log('Getting MCP token...');
      const mcpToken = await this.getMcpToken();
      console.log('MCP token received:', mcpToken.substring(0, 8) + '...');
      
      // Connect to MCP WebSocket with token
      console.log('Connecting to WebSocket with authentication...');
      await this.mcpClient.connectWithToken(mcpToken);
      console.log('Authenticated WebSocket connection successful');
      
      // Initialize MCP session
      console.log('Initializing MCP session...');
      await this.mcpClient.initialize();
      console.log('MCP session initialized');
      
      console.log('=== AUTHENTICATED MCP CONNECTION SUCCESSFUL ===');
    } catch (error) {
      console.error('=== AUTHENTICATED MCP CONNECTION FAILED ===', error);
      
      // Update connection state
      this.mcpConnectionState = {
        connected: false,
        authenticated: false,
        error: `Auth connection failed: ${error}`
      };
      
      throw error;
    }
  }

  private async initializeMcpConnectionWithoutAuth(): Promise<void> {
    console.log('=== INITIALIZING UNAUTHENTICATED MCP CONNECTION ===');
    
    try {
      // Connect to MCP WebSocket without token (for public tools)
      console.log('Connecting to WebSocket without authentication...');
      await this.mcpClient.connectWithoutAuth();
      console.log('Unauthenticated WebSocket connection successful');
      
      // Initialize MCP session
      console.log('Initializing MCP session...');
      await this.mcpClient.initialize();
      console.log('MCP session initialized');
      
      console.log('=== UNAUTHENTICATED MCP CONNECTION SUCCESSFUL ===');
    } catch (error) {
      console.error('=== UNAUTHENTICATED MCP CONNECTION FAILED ===', error);
      
      // Update connection state
      this.mcpConnectionState = {
        connected: false,
        authenticated: false,
        error: `Public connection failed: ${error}`
      };
      
      throw error;
    }
  }


  private async getMcpToken(): Promise<string> {
    try {
      // Get current auth token from localStorage
      const authToken = localStorage.getItem('auth_token');
      
      if (!authToken) {
        throw new Error('No authentication token available');
      }
      
      // Call backend to get MCP token
      const response = await fetch('http://localhost:8000/auth/get-mcp-token', {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to get MCP token: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('MCP token obtained successfully');
      return data.mcp_token;
    } catch (error) {
      console.error('Failed to get MCP token:', error);
      throw error;
    }
  }

  private async processMcpQuery(query: string): Promise<void> {
    this.isMcpProcessing = true;
    
    // Add processing message with action tracking
    const processingMessage: ChatMessage = {
      text: 'Processing with MCP...',
      sender: 'bot',
      streaming: false,
      actionEvents: []
    };
    this.messages.push(processingMessage);
    
    try {
      // Track action: Tool prediction
      this.addActionEvent(processingMessage, 'tool_prediction', { status: 'starting' });
      
      // Determine which tool to call based on query content
      const toolCall = this.predictMcpTool(query);
      console.log('Tool call to be made:', toolCall);
      
      this.addActionEvent(processingMessage, 'tool_prediction', { 
        status: 'completed',
        predicted_tool: toolCall.name,
        arguments: toolCall.arguments 
      });
      
      // Track action: WebSocket connection check
      this.addActionEvent(processingMessage, 'connection_check', { 
        connected: this.mcpConnectionState.connected,
        authenticated: this.mcpConnectionState.authenticated 
      });
      
      // Track action: Tool execution start
      this.addActionEvent(processingMessage, 'tool_execution_start', { 
        tool_name: toolCall.name,
        arguments: toolCall.arguments 
      });
      
      // Call the MCP tool
      const result = await this.mcpClient.callTool(toolCall);
      console.log('MCP tool result:', result);
      
      // Track action: Tool execution complete
      this.addActionEvent(processingMessage, 'tool_execution_complete', { 
        tool_name: toolCall.name,
        success: true,
        result_preview: this.getResultPreview(result)
      });
      
      // Update the processing message with result
      const resultText = this.formatMcpResult(result);
      processingMessage.text = resultText;
      processingMessage.streaming = false;
      
    } catch (error) {
      console.error('MCP query failed:', error);
      
      // Track action: Error
      this.addActionEvent(processingMessage, 'error', { 
        error_message: error?.toString() || 'Unknown error',
        error_type: error?.constructor?.name || 'Error'
      });
      
      // Check if this is an authentication error
      const errorMessage = error?.toString() || '';
      if (errorMessage.includes('Authentication required')) {
        // Show authentication required message with sign-in button (same as streaming)
        processingMessage.text = 'Error: Authentication required';
        processingMessage.error = true;
        processingMessage.requiresAuth = true;
      } else {
        // Show generic error
        processingMessage.text = `MCP Error: ${error}`;
        processingMessage.error = true;
      }
    } finally {
      this.isMcpProcessing = false;
    }
  }

  private predictMcpTool(query: string): { name: string; arguments: any } {
    const lowerQuery = query.toLowerCase();
    console.log('=== PREDICTING MCP TOOL ===');
    console.log('Original query:', query);
    console.log('Lowercase query:', lowerQuery);
    
    // Simple query prediction logic
    if (lowerQuery.includes('inventory') || lowerQuery.includes('stock') || lowerQuery.includes('available') || lowerQuery.includes('item')) {
      // Extract item ID from query - prioritize pure numbers first
      let itemId = 'UNKNOWN';
      
      // First, look for pure numeric IDs (most common case)
      const numericMatch = query.match(/\b\d+\b/);
      console.log('Pure numeric matches:', numericMatch);
      
      if (numericMatch) {
        itemId = numericMatch[0];
        console.log('Found numeric item ID:', itemId);
      } else {
        // Fallback: look for alphanumeric codes (like "ABC123")
        const alphanumericMatch = query.match(/\b[A-Z0-9]{2,}\b/i);
        console.log('Alphanumeric matches:', alphanumericMatch);
        
        if (alphanumericMatch) {
          // Filter out common English words
          const commonWords = /^(can|you|the|and|are|item|of|details|share|availability|status|what|how|is|in|with|for|to|from)$/i;
          const validMatch = alphanumericMatch.find(match => !commonWords.test(match));
          itemId = validMatch || 'UNKNOWN';
          console.log('Found alphanumeric item ID:', itemId);
        }
      }
      
      console.log('Final selected item ID:', itemId);
      
      return {
        name: 'lookup_inventory',
        arguments: {
          item_id: itemId
        }
      };
    } else if (lowerQuery.includes('order') || lowerQuery.includes('status')) {
      const orderIdMatch = query.match(/\b\d{3,}\b/);
      console.log('Order ID match:', orderIdMatch);
      return {
        name: 'check_order_status',
        arguments: {
          order_id: orderIdMatch ? orderIdMatch[0] : 'UNKNOWN'
        }
      };
    } else if (lowerQuery.includes('refund')) {
      const refundIdMatch = query.match(/\b\d{3,}\b/);
      console.log('Refund ID match:', refundIdMatch);
      return {
        name: 'track_refund',
        arguments: {
          refund_id: refundIdMatch ? refundIdMatch[0] : 'UNKNOWN'
        }
      };
    }
    
    // Default to inventory lookup with generic help
    console.log('No specific pattern matched, defaulting to inventory lookup');
    return {
      name: 'lookup_inventory',
      arguments: {
        item_id: 'HELP'
      }
    };
  }

  private addActionEvent(message: ChatMessage, type: string, data: any): void {
    if (!message.actionEvents) {
      message.actionEvents = [];
    }
    
    const actionEvent: ActionEvent = {
      type,
      timestamp: new Date().toISOString(),
      data
    };
    
    message.actionEvents.push(actionEvent);
    console.log('Added action event:', actionEvent);
  }

  private getResultPreview(result: any): string {
    if (result && result.content && Array.isArray(result.content)) {
      const text = result.content.map((item: any) => item.text || item).join('\n');
      return text.length > 100 ? text.substring(0, 97) + '...' : text;
    }
    
    if (typeof result === 'string') {
      return result.length > 100 ? result.substring(0, 97) + '...' : result;
    }
    
    const jsonStr = JSON.stringify(result, null, 2);
    return jsonStr.length > 100 ? jsonStr.substring(0, 97) + '...' : jsonStr;
  }

  private formatMcpResult(result: any): string {
    if (result && result.content && Array.isArray(result.content)) {
      return result.content.map((item: any) => item.text || item).join('\n');
    }
    
    if (typeof result === 'string') {
      return result;
    }
    
    return JSON.stringify(result, null, 2);
  }

  private handleStreamingResponse(response: StreamingResponse): void {
    if (response.type === 'event' && response.event && this.currentStreamingMessage) {
      const event = response.event;
      
      // Add event to message history
      if (!this.currentStreamingMessage.streamEvents) {
        this.currentStreamingMessage.streamEvents = [];
      }
      this.currentStreamingMessage.streamEvents.push(event);
      
      // Update message content based on event type
      switch (event.type) {
        case 'prediction':
          this.currentStreamingMessage.text = `Analyzing query... Predicted tools: ${event.data.predicted_tools?.join(', ') || 'none'}`;
          break;
          
        case 'auth_check':
          if (event.data.required && !event.data.valid) {
            this.currentStreamingMessage.text = 'Authentication required for this request.';
            this.currentStreamingMessage.error = true;
            this.currentStreamingMessage.requiresAuth = true;
          } else {
            this.currentStreamingMessage.text = 'Authentication verified. Processing request...';
          }
          break;
          
        case 'tool_execution_start':
          this.currentStreamingMessage.text = `Executing: ${event.data.tool_name}...`;
          if (this.currentStreamingMessage.progress) {
            this.currentStreamingMessage.progress = {
              step: event.data.step || 0,
              total: event.data.total_steps || 0,
              current: `Running ${event.data.tool_name}`,
              percentage: 0
            };
          }
          break;
          
        case 'tool_execution_complete':
          this.currentStreamingMessage.text = `Completed: ${event.data.tool_name}`;
          if (this.currentStreamingMessage.progress) {
            const percentage = event.data.step && event.data.total_steps ? 
              Math.round((event.data.step / event.data.total_steps) * 100) : 0;
            this.currentStreamingMessage.progress = {
              step: event.data.step || 0,
              total: event.data.total_steps || 0,
              current: `Completed ${event.data.tool_name}`,
              percentage
            };
          }
          break;
          
        case 'progress':
          if (this.currentStreamingMessage.progress) {
            this.currentStreamingMessage.progress = {
              step: event.data.completed_steps || 0,
              total: event.data.total_steps || 0,
              current: event.data.current_step || '',
              percentage: event.data.percentage || 0
            };
          }
          break;
          
        case 'final_answer':
          this.currentStreamingMessage.text = event.data.response || 'Request completed.';
          this.currentStreamingMessage.streaming = false;
          if (this.currentStreamingMessage.progress) {
            this.currentStreamingMessage.progress.percentage = 100;
            this.currentStreamingMessage.progress.current = 'Completed';
          }
          break;
          
        case 'error':
          this.currentStreamingMessage.text = `Error: ${event.data.error}`;
          this.currentStreamingMessage.error = true;
          this.currentStreamingMessage.streaming = false;
          break;
      }
    }
  }
  
  private handleStreamingError(error: any): void {
    this.isStreaming = false;
    if (this.currentStreamingMessage) {
      this.currentStreamingMessage.text = `Error: ${error.message || 'An unexpected error occurred'}`;
      this.currentStreamingMessage.error = true;
      this.currentStreamingMessage.streaming = false;
    }
    this.currentStreamingMessage = null;
  }
  
  private handleStreamingComplete(): void {
    this.isStreaming = false;
    if (this.currentStreamingMessage) {
      this.currentStreamingMessage.streaming = false;
    }
    this.currentStreamingMessage = null;
  }
  
  private getConversationHistory(): any[] {
    // Convert recent messages to conversation history format
    return this.messages
      .filter(msg => !msg.streaming && !msg.error)
      .slice(-10) // Last 10 messages
      .map(msg => ({
        role: msg.sender === 'user' ? 'user' : 'assistant',
        content: msg.text
      }));
  }

  onSignInClick(): void {
    // Get the authentication URL (from session storage or default)
    const authUrl = this.configService.getAuthUrl();
    // Open authentication URL in a new tab
    window.open(authUrl, '_blank');
  }

  private scrollToBottom(): void {
    try {
      this.messageContainer.nativeElement.scrollTop = this.messageContainer.nativeElement.scrollHeight;
    } catch (err) { }
  }

  private loadInputHistory(): void {
    try {
      const storedHistory = sessionStorage.getItem(this.HISTORY_KEY);
      if (storedHistory) {
        this.inputHistory = JSON.parse(storedHistory);
      }
    } catch (error) {
      console.warn('Failed to load input history from session storage:', error);
      this.inputHistory = [];
    }
  }

  private saveInputHistory(): void {
    try {
      sessionStorage.setItem(this.HISTORY_KEY, JSON.stringify(this.inputHistory));
    } catch (error) {
      console.warn('Failed to save input history to session storage:', error);
    }
  }

  private addToHistory(query: string): void {
    if (query && query.trim()) {
      // Remove the query if it already exists to avoid duplicates
      const existingIndex = this.inputHistory.indexOf(query);
      if (existingIndex !== -1) {
        this.inputHistory.splice(existingIndex, 1);
      }
      
      // Add to the beginning of the history
      this.inputHistory.unshift(query);
      
      // Limit history to 50 items
      if (this.inputHistory.length > 50) {
        this.inputHistory = this.inputHistory.slice(0, 50);
      }
      
      this.saveInputHistory();
    }
  }

  private navigateHistory(direction: 'up' | 'down'): void {
    if (this.inputHistory.length === 0) return;
    
    if (direction === 'up') {
      if (this.historyIndex < this.inputHistory.length - 1) {
        this.historyIndex++;
        this.queryText = this.inputHistory[this.historyIndex];
        this.queryTextChange.emit(this.queryText);
      }
    } else if (direction === 'down') {
      if (this.historyIndex > 0) {
        this.historyIndex--;
        this.queryText = this.inputHistory[this.historyIndex];
        this.queryTextChange.emit(this.queryText);
      } else if (this.historyIndex === 0) {
        this.historyIndex = -1;
        this.queryText = '';
        this.queryTextChange.emit(this.queryText);
      }
    }
  }

  private resetHistoryIndex(): void {
    this.historyIndex = -1;
  }

  hasAuthToken(): boolean {
    return !!localStorage.getItem('auth_token');
  }

  getMcpButtonText(): string {
    if (this.mcpConnectionState.connected && this.mcpConnectionState.authenticated) {
      return 'Send (MCP) ✓ Auth';
    }
    
    if (this.mcpConnectionState.connected && !this.mcpConnectionState.authenticated) {
      return 'Send (MCP) ✓ Public';
    }
    
    if (this.mcpConnectionState.error) {
      return 'Send (MCP) ✗';
    }
    
    return 'Send (MCP)';
  }

  getMcpButtonTooltip(): string {
    if (this.mcpConnectionState.connected && this.mcpConnectionState.authenticated) {
      return 'Connected to MCP server with authentication - all tools available';
    }
    
    if (this.mcpConnectionState.connected && !this.mcpConnectionState.authenticated) {
      return 'Connected to MCP server - public tools only (inventory lookup)';
    }
    
    if (this.mcpConnectionState.error) {
      return `MCP connection error: ${this.mcpConnectionState.error}`;
    }
    
    if (!this.hasAuthToken()) {
      return 'Will connect for public tools (inventory). Sign in for order/refund tools.';
    }
    
    return 'Direct MCP tool execution';
  }
}
