import { Component } from '@angular/core';
import { WeatherAdvisorComponent } from './weather-advisor-component/weather-advisor-component';
import { provideHttpClient } from '@angular/common/http';


@Component({
  selector: 'app-root',
  imports: [ WeatherAdvisorComponent],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  protected title = 'Weather Advisor';
}
