// AuthContext.js
import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [auth, setAuth] = useState(() => {
    const storedToken = localStorage.getItem('authToken');
    return { token: !!storedToken }; // Convert storedToken (string) to boolean
  });
  useEffect(() => {
    const storedToken = localStorage.getItem('authToken');
    if (storedToken) {
      setAuth({ token: true });
      console.log('AuthProvider: Token found in localStorage, setting auth to true');
    } else {
      console.log('AuthProvider: No token found in localStorage, setting auth to false');
    }
  }, []);

  const login = (token) => {
    setAuth({ token: true });
    localStorage.setItem('authToken', token);
    console.log('Login: Token set and state updated');
  };

  const logout = () => {
    setAuth({ token: false });
    localStorage.removeItem('authToken');
    console.log('Logout: Token removed and state updated');
  };

  useEffect(() => {
    console.log('Auth State Changed:', auth);
  }, [auth]);

  return (
    <AuthContext.Provider value={{ auth, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
