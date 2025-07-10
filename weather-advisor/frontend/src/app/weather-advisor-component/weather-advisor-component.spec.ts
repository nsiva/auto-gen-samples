import { ComponentFixture, TestBed } from '@angular/core/testing';

import { WeatherAdvisorComponent } from './weather-advisor-component';

describe('WeatherAdvisorComponent', () => {
  let component: WeatherAdvisorComponent;
  let fixture: ComponentFixture<WeatherAdvisorComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [WeatherAdvisorComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(WeatherAdvisorComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
