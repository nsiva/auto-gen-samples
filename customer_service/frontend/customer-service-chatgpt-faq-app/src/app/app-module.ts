import { CommonModule } from '@angular/common';

import { NgModule, provideBrowserGlobalErrorListeners } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms'; // Import FormsModule
import { RouterModule } from '@angular/router';

import { App } from './app';
import { ChatInterface } from './components/chat-interface/chat-interface';
import { FaqSection } from './components/faq-section/faq-section';
import { routes } from './app.routes';

@NgModule({
  declarations: [
    App
  ],
  imports: [
    CommonModule,
    BrowserModule,
    FormsModule,
    RouterModule.forRoot(routes),
    ChatInterface,
    FaqSection
  ],
  providers: [
    provideBrowserGlobalErrorListeners()
  ],
  bootstrap: [App]
})
export class AppModule { }
