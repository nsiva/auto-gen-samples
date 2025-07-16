import { Component, Input, Output, EventEmitter, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

export interface ChatMessage {
  text: string;
  sender: 'user' | 'bot';
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

  constructor() { }

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

  private scrollToBottom(): void {
    try {
      this.messageContainer.nativeElement.scrollTop = this.messageContainer.nativeElement.scrollHeight;
    } catch (err) { }
  }
}
