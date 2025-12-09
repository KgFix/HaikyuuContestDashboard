import React, { useEffect } from 'react';
import { useAuth } from './AuthContext';
import { useNavigate, useLocation } from 'react-router-dom';

export const AuthCallback: React.FC = () => {
  const { setAuthData } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const handleCallback = async () => {
      // Parse token from URL
      const params = new URLSearchParams(location.search);
      const token = params.get('token');

      if (!token) {
        console.error('No token in callback URL');
        navigate('/');
        return;
      }

      try {
        // Fetch user info with the token
        const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
        const response = await fetch(`${apiBaseUrl}/api/auth/me`, {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });

        if (!response.ok) {
          throw new Error('Failed to fetch user info');
        }

        const userInfo = await response.json();

        // Save auth data
        setAuthData(token, userInfo);

        // Redirect to dashboard
        navigate('/', { replace: true });
      } catch (error) {
        console.error('Authentication callback error:', error);
        navigate('/');
      }
    };

    handleCallback();
  }, [location, setAuthData, navigate]);

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '100vh',
      flexDirection: 'column',
      gap: '1rem'
    }}>
      <div className="spinner"></div>
      <p>Completing authentication...</p>
    </div>
  );
};
