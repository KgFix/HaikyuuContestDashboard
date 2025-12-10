import React, { useEffect } from 'react';
import { useAuth } from './AuthContext';
import { useNavigate, useLocation } from 'react-router-dom';

export const AuthCallback: React.FC = () => {
  const { setAuthData } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const handleCallback = async () => {
      console.log('AuthCallback: Starting authentication callback');
      console.log('AuthCallback: Current URL:', window.location.href);
      
      // Parse token from URL
      const params = new URLSearchParams(location.search);
      const token = params.get('token');

      console.log('AuthCallback: Token present:', !!token);

      if (!token) {
        console.error('AuthCallback: No token in callback URL');
        alert('Authentication failed: No token received');
        navigate('/login');
        return;
      }

      try {
        // Fetch user info with the token
        const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
        console.log('AuthCallback: API Base URL:', apiBaseUrl);
        console.log('AuthCallback: Fetching user info...');
        
        const response = await fetch(`${apiBaseUrl}/api/auth/me`, {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });

        console.log('AuthCallback: Response status:', response.status);

        if (!response.ok) {
          const errorText = await response.text();
          console.error('AuthCallback: API error:', errorText);
          throw new Error(`Failed to fetch user info: ${response.status}`);
        }

        const userInfo = await response.json();
        console.log('AuthCallback: User info received:', userInfo);

        // Save auth data
        setAuthData(token, userInfo);
        console.log('AuthCallback: Auth data saved, redirecting to dashboard');

        // Redirect to dashboard
        navigate('/', { replace: true });
      } catch (error) {
        console.error('Authentication callback error:', error);
        alert(`Authentication failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        navigate('/login');
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
