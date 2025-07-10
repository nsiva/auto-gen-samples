import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

import { WeatherService } from '../weather-service';

@Component({
  selector: 'app-weather-advisor',
  standalone: true,
  imports: [FormsModule,CommonModule],
  templateUrl: './weather-advisor-component.html',
  styleUrl: './weather-advisor-component.css'
})
export class WeatherAdvisorComponent {
  zipCode = '';
  result: any = null;
  error: string | null = null;

  constructor(private weatherService: WeatherService) {
    console.log('WeatherAdvisorComponent constructor called');
  }

  
  checkZip() {
    this.error = null;
    this.result = null;
    console.log('Checking umbrella for zip code:', this.zipCode);
    this.weatherService.checkUmbrella(this.zipCode).subscribe({
      next: (res) => {
        console.log('API Response:', res);
        this.result = res;
      },
      error: (err) => {
        console.error('API Error:', err);
        this.error = err.error?.detail || 'An error occurred';
      },
    });
    // this.result = {"weather": "Cloudy with a chance of rain",
    //   "advice":"Take umbrella"}; // Mock response for demonstration
    // this.weatherService.checkUmbrella(this.zipCode).subscribe({
    //   next: (res) => {
    //     this.result = res;
    //     console.log('Response received:', this.result);
    //   },
    //   error: (err) => {
    //     this.error = err.error.detail || 'An error occurred';
    //     console.error('Error occurred:', this.error);
    //   }
    // });
    //console.log('Result:', this.result);
    // Assuming weatherService is injected properly
    // Uncomment the following lines when WeatherService is properly injected
    //var response = this.weatherService.checkUmbrella(this.zipCode);
    // this.weatherService.checkUmbrella(this.zipCode).subscribe({
    //   next: (res) => (this.result = res),
    //   error: (err) => (this.error = err.error.detail || 'An error occurred'),
    // });
  }
}
