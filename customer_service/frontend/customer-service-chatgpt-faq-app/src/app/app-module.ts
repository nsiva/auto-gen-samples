import { CommonModule } from '@angular/common';

import { NgModule, provideBrowserGlobalErrorListeners } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms'; // Import FormsModule

import { App } from './app';
import { ChatInterface } from './components/chat-interface/chat-interface';
import { FaqSection } from './components/faq-section/faq-section';

@NgModule({
  declarations: [
    App
  ],
  imports: [
    CommonModule,
    BrowserModule,
    FormsModule,
    ChatInterface,
    FaqSection
  ],
  providers: [
    provideBrowserGlobalErrorListeners()
  ],
  bootstrap: [App]
})
export class AppModule { }
