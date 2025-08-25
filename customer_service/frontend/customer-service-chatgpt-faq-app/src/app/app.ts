import { Component } from '@angular/core';
import { ChatMessage } from './components/chat-interface/chat-interface';
import { FaqCategory } from './models/faq.model';

@Component({
  selector: 'app-root',
  templateUrl: './app.html',
  standalone: false,
  styleUrl: './app.scss'
})
export class App {
  protected title = 'Customer Service Chatbot ';

  chatMessages: ChatMessage[] = [];
  currentQueryText: string = ''; // Text for the input box

  faqCategories: FaqCategory[] = [];

  ngOnInit(): void {
    this.loadFaqData();
    this.addInitialBotMessage();
  }

  private addInitialBotMessage(): void {
    this.chatMessages.push({
      text: "Hello! How can I assist you today? Feel free to ask a question or select from the FAQs.",
      sender: 'bot'
    });
  }

  private loadFaqData(): void {
    this.faqCategories = [
      {
        name: 'Order/Item Information',
        isExpanded: true,
        faqs: [
          { question: 'What is the status of order 123?', answer: 'We provide AI-powered assistance for various queries.' },
          { question: 'Can you share the availability details of item 123?', answer: 'Might be available' },
          { question: 'What is the refund status for order 456?', answer: 'God knows' }
        ]
      },
      {
        name: 'Account & Billing',
        isExpanded: false,
        faqs: [
          { question: 'How do I reset my password?', answer: 'You can reset your password from the account settings page.' },
          { question: 'Where can I see my billing history?', answer: 'Billing history is available under your user profile.' },
          { question: 'How do I cancel my subscription?', answer: 'Please contact support to cancel your subscription.' }
        ]
      },
      {
        name: 'Technical Support',
        isExpanded: false,
        faqs: [
          { question: 'I cannot log in.', answer: 'Please check your credentials or try resetting your password.' },
          { question: 'The chat interface is not loading.', answer: 'Try refreshing the page or clearing your browser cache.' },
          { question: 'How to report a bug?', answer: 'You can report bugs via our feedback form or contact support.' }
        ]
      },
      {
        name: 'Privacy & Security',
        isExpanded: false,
        faqs: [
          { question: 'What is your privacy policy?', answer: 'Our privacy policy is available on our website.' },
          { question: 'Is my data secure?', answer: 'Yes, we use industry-standard encryption for data security.' },
          { question: 'Do you share my data with third parties?', answer: 'We do not share your personal data without explicit consent.' }
        ]
      }
    ];
  }

  async handleQuerySubmit(query: string): Promise<void> {
    
    // --- Core Change: Prepare context for API call ---
    const conversationContext = this.chatMessages.map(msg => ({
      role: msg.sender === 'user' ? 'user' : 'assistant', // LLMs typically use 'user'/'assistant' roles
      content: msg.text
    }));

    // Add user message to chat
    this.chatMessages.push({ text: query, sender: 'user' });

    // Call FAQ API and show response
    const faqAnswer = await this.invokeChatWithFaq(query, conversationContext);
    let botMessage: ChatMessage;
    
    if (faqAnswer === 'AUTH_REQUIRED') {
      botMessage = {
        text: 'Authentication required for this request.',
        sender: 'bot',
        error: true,
        requiresAuth: true
      };
    } else if (faqAnswer) {
      botMessage = {
        text: `I received your question: "${query}". Thank you!\n\nAnswer: ${faqAnswer}`,
        sender: 'bot'
      };
    } else {
      botMessage = {
        text: `I received your question: "${query}". Thank you!\n\nI'm still learning and might not have a direct answer for this yet. How else can I help?`,
        sender: 'bot'
      };
    }
    
    this.chatMessages.push(botMessage);
  }
  //  private callBackendWithContext(newQuery: string, context: { role: string, content: string }[]): void {

//    console.log("Sending to backend with context:");
  async invokeChatWithFaq(question: string, context: {role: string, content: string}[]): Promise<string> {

    console.log("New Query:", question);
    console.log("Full Context (History):", context);

    this.currentQueryText = question; // Populate the input box with the FAQ question
    const payload = { query: question.trim(), history: context  }; // Include context in the payload
    
    try {
      const response = await fetch('http://127.0.0.1:8000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (response.status === 401) {
        // Check if auth URL is provided in response headers
        const authUrl = response.headers.get('X-Auth-URL');
        if (authUrl) {
          sessionStorage.setItem('auth-url', authUrl);
        }
        return 'AUTH_REQUIRED';
      }
      const data = await response.json();
      return data.response || 'No response';
    } catch (error: any) {
      return 'Encountered a Server Error: ' + error.message;
    }
  }

  handleFaqQuestionSelected(question: string): void {
    this.currentQueryText = question; // Populate the input box
  }

  handleQueryTextChange(newText: string): void {
    this.currentQueryText = newText; // Keep track of current input for two-way binding
  }

  private findMatchingFaqAnswer(query: string): string | null {
    const lowerQuery = query.toLowerCase();
    for (const category of this.faqCategories) {
      for (const faq of category.faqs) {
        if (faq.question.toLowerCase() === lowerQuery) {
          return faq.answer;
        }
      }
    }
    return null;
  }
}
