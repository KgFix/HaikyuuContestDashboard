import React from 'react';
import { DiscordLogin } from '../auth/DiscordLogin';
import './Login.css';

export const Login: React.FC = () => {
  return (
    <div className="login-page">
      <div className="login-container">
        <div className="login-header">
          <h1>🏐 Haikyuu Contest Dashboard</h1>
          <p>Track your club's performance and compete with others</p>
        </div>
        
        <div className="login-content">
          <div className="login-info">
            <h2>Welcome!</h2>
            <p>Sign in with Discord to access your club's dashboard and view:</p>
            <ul>
              <li>📊 Club performance history</li>
              <li>👥 Team activity tracking</li>
              <li>🏆 Player leaderboards</li>
              <li>📈 Progress over time</li>
            </ul>
          </div>
          
          <div className="login-action">
            <DiscordLogin />
            <p className="login-note">
              You'll be redirected to Discord to authorize access
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
