import { Injectable } from '@angular/core';
import { Observable, Subject, BehaviorSubject } from 'rxjs';
import { StreamEvent, StreamingResponse, ToolProgress } from '../models/streaming.models';

@Injectable({
  providedIn: 'root'
})
export class StreamingQueryService {
  private eventSubject = new Subject<StreamEvent>();
  private progressSubject = new BehaviorSubject<ToolProgress>({ step: 0, total: 0, current: '', percentage: 0 });
  private isStreamingSubject = new BehaviorSubject<boolean>(false);

  // Observable streams for components to subscribe to
  public events$ = this.eventSubject.asObservable();
  public progress$ = this.progressSubject.asObservable();
  public isStreaming$ = this.isStreamingSubject.asObservable();

  constructor() {}

  /**
   * Submit a query to the streaming endpoint and return observable of stream events
   */
  async submitStreamingQuery(query: string, history: any[] = []): Promise<Observable<StreamingResponse>> {
    const responseSubject = new Subject<StreamingResponse>();
    
    try {
      this.isStreamingSubject.next(true);
      this.resetProgress();

      const response = await fetch('http://localhost:8000/ask-stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/plain',
        },
        body: JSON.stringify({
          query: query,
          history: history,
          include_intermediate: true,
          include_progress: true
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Handle Server-Sent Events
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('No response body reader available');
      }

      const processStream = async () => {
        try {
          while (true) {
            const { done, value } = await reader.read();
            
            if (done) {
              this.isStreamingSubject.next(false);
              responseSubject.complete();
              break;
            }

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const eventData = line.substring(6); // Remove 'data: ' prefix
                  if (eventData.trim()) {
                    const streamEvent = JSON.parse(eventData) as StreamEvent;
                    this.handleStreamEvent(streamEvent);
                    
                    // Emit the processed event
                    responseSubject.next({
                      type: 'event',
                      event: streamEvent,
                      timestamp: new Date()
                    });
                  }
                } catch (parseError) {
                  console.warn('Failed to parse stream event:', parseError, line);
                }
              }
            }
          }
        } catch (streamError) {
          console.error('Stream processing error:', streamError);
          this.isStreamingSubject.next(false);
          responseSubject.error(streamError);
        }
      };

      // Start processing the stream
      processStream();

    } catch (error) {
      console.error('Streaming query error:', error);
      this.isStreamingSubject.next(false);
      responseSubject.error(error);
    }

    return responseSubject.asObservable();
  }

  /**
   * Handle individual stream events and update internal state
   */
  private handleStreamEvent(event: StreamEvent): void {
    this.eventSubject.next(event);

    switch (event.type) {
      case 'prediction':
        console.log('Prediction received:', event.data);
        break;

      case 'auth_check':
        console.log('Auth check:', event.data);
        break;

      case 'tool_execution_start':
        this.updateProgress({
          step: event.data.step || 0,
          total: event.data.total_steps || 0,
          current: `Starting ${event.data.tool_name}`,
          percentage: 0
        });
        break;

      case 'tool_execution_complete':
        this.updateProgress({
          step: event.data.step || 0,
          total: event.data.total_steps || 0,
          current: `Completed ${event.data.tool_name}`,
          percentage: event.data.step ? Math.round((event.data.step / (event.data.total_steps || 1)) * 100) : 0
        });
        break;

      case 'progress':
        if (event.data.percentage !== undefined) {
          this.updateProgress({
            step: event.data.completed_steps || 0,
            total: event.data.total_steps || 0,
            current: event.data.current_step || '',
            percentage: event.data.percentage
          });
        }
        break;

      case 'final_answer':
        this.updateProgress({
          step: this.progressSubject.value.total,
          total: this.progressSubject.value.total,
          current: 'Completed',
          percentage: 100
        });
        break;

      case 'error':
        console.error('Stream error:', event.data);
        break;

      default:
        console.log('Unknown event type:', event.type, event.data);
    }
  }

  /**
   * Update progress information
   */
  private updateProgress(progress: ToolProgress): void {
    this.progressSubject.next(progress);
  }

  /**
   * Reset progress to initial state
   */
  private resetProgress(): void {
    this.progressSubject.next({
      step: 0,
      total: 0,
      current: 'Initializing...',
      percentage: 0
    });
  }

  /**
   * Get current progress state
   */
  getCurrentProgress(): ToolProgress {
    return this.progressSubject.value;
  }

  /**
   * Check if currently streaming
   */
  isStreaming(): boolean {
    return this.isStreamingSubject.value;
  }

  /**
   * Cancel current streaming operation
   */
  cancelStreaming(): void {
    this.isStreamingSubject.next(false);
    this.resetProgress();
  }
}