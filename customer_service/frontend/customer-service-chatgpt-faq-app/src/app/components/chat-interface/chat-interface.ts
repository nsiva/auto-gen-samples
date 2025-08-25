import { Component, Input, Output, EventEmitter, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { StreamingQueryService } from '../../services/streaming-query.service';
import { ConfigService } from '../../services/config.service';
import { StreamEvent, StreamingResponse, ToolProgress, ChatMessage as StreamingChatMessage } from '../../models/streaming.models';

export interface ChatMessage {
  text: string;
  sender: 'user' | 'bot';
  streaming?: boolean;
  progress?: ToolProgress;
  streamEvents?: StreamEvent[];
  error?: boolean;
  requiresAuth?: boolean;
}

@Component({
  selector: 'app-chat-interface',
  templateUrl: './chat-interface.html',
  styleUrls: ['./chat-interface.scss'],
  standalone: true,
  imports: [CommonModule, FormsModule]
})
export class ChatInterface implements AfterViewChecked {
  @Input() messages: ChatMessage[] = [];
  @Input() queryText: string = ''; // Input for pre-filling from FAQ
  @Output() querySubmit = new EventEmitter<string>();
  @Output() queryTextChange = new EventEmitter<string>(); // For two-way binding or direct update

  @ViewChild('messageContainer') private messageContainer!: ElementRef;

  // Streaming properties
  public isStreaming = false;
  public streamingProgress: ToolProgress = { step: 0, total: 0, current: '', percentage: 0 };
  public currentStreamingMessage: ChatMessage | null = null;

  constructor(
    private streamingQueryService: StreamingQueryService,
    private configService: ConfigService
  ) { }

  ngAfterViewChecked() {
    this.scrollToBottom();
  }

  onSendClick(): void {
    if (this.queryText.trim()) {
      this.querySubmit.emit(this.queryText.trim());
      this.queryText = ''; // Clear input after sending
      this.queryTextChange.emit(''); // Notify parent about cleared text
    }
  }

  async onSendStreamingClick(): Promise<void> {
    if (this.queryText.trim() && !this.isStreaming) {
      const userQuery = this.queryText.trim();
      
      // Add user message to chat
      const userMessage: ChatMessage = {
        text: userQuery,
        sender: 'user'
      };
      this.messages.push(userMessage);
      
      // Clear input
      this.queryText = '';
      this.queryTextChange.emit('');
      
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
}
