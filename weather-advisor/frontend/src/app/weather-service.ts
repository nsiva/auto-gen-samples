import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http'; // Import HttpClient
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class WeatherService {
  private apiUrl = 'http://localhost:8000/should_take_umbrella'; // Local server URL

  constructor(private http: HttpClient) {
    console.log('WeatherService initialized');
  }

  checkUmbrella(zipCode: string): Observable<any> {
    // Make an HTTP POST request to the local server
    return this.http.post(this.apiUrl, { zip_code: zipCode });
  }
}
