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
  protected title = 'customer-service-chatgpt-faq-app';

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
        name: 'General Information',
        isExpanded: true,
        faqs: [
          { question: 'What is your service?', answer: 'We provide AI-powered assistance for various queries.' },
          { question: 'How do I get started?', answer: 'You can start by typing your question in the chat box.' },
          { question: 'Is this service free?', answer: 'Basic usage is free, premium features may vary.' }
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

  handleQuerySubmit(query: string): void {
    // Add user message to chat
    this.chatMessages.push({ text: query, sender: 'user' });

    // Simulate a bot response (replace with actual API call)
    setTimeout(() => {
      let botResponse = `I received your question: "${query}". Thank you!`;
      const matchedFaq = this.findMatchingFaqAnswer(query);
      if (matchedFaq) {
        botResponse += `\n\nFAQ Answer: ${matchedFaq}`;
      } else {
        botResponse += "\n\nI'm still learning and might not have a direct answer for this yet. How else can I help?";
      }
      this.chatMessages.push({ text: botResponse, sender: 'bot' });
    }, 500);
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
