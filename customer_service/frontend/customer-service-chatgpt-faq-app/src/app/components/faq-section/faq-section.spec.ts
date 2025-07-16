import { ComponentFixture, TestBed } from '@angular/core/testing';

import { FaqSection } from './faq-section';

describe('FaqSection', () => {
  let component: FaqSection;
  let fixture: ComponentFixture<FaqSection>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [FaqSection]
    })
    .compileComponents();

    fixture = TestBed.createComponent(FaqSection);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
