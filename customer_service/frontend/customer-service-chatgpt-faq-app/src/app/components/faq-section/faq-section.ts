import { Component, Output, EventEmitter, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FaqCategory } from '../../models/faq.model';

@Component({
  selector: 'app-faq-section',
  templateUrl: './faq-section.html',
  styleUrls: ['./faq-section.scss'],
  standalone: true,
  imports: [CommonModule]
})
export class FaqSection {
  @Input() categories: FaqCategory[] = [];
  @Output() questionSelected = new EventEmitter<string>();

  constructor() { }

  toggleCategory(category: FaqCategory): void {
    category.isExpanded = !category.isExpanded;
  }

  selectQuestion(question: string): void {
    this.questionSelected.emit(question);
  }
}