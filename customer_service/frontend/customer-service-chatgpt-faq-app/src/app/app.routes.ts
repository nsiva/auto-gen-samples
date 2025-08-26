import { Routes } from '@angular/router';
import { AuthCallbackComponent } from './components/auth-callback/auth-callback';

export const routes: Routes = [
  {
    path: 'auth/callback',
    component: AuthCallbackComponent,
    title: 'Authentication Callback'
  },
  {
    path: '',
    redirectTo: '/',
    pathMatch: 'full'
  }
];