// PrivateRoute.js
import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from './AuthContext';

const PrivateRoute = () => {
  const { auth } = useAuth();

  return auth.token ? <Outlet /> : <Navigate to="/login" replace />;
};

export default PrivateRoute;
