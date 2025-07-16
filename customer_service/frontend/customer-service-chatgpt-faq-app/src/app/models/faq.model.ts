// src/app/models/faq.model.ts
export interface FaqItem {
  question: string;
  answer: string; // Not directly used in the query box, but good for data integrity
}

export interface FaqCategory {
  name: string;
  faqs: FaqItem[];
  isExpanded: boolean; // To control collapse/expand state
}