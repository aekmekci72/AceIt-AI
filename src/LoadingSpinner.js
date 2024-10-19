import React from 'react';

const LoadingDots = () => {
  return (
    <div className="flex items-center justify-center space-x-2">
      {[0, 1, 2].map((index) => (
        <div
          key={index}
          className="w-3 h-3 bg-cyan-500 rounded-full animate-bounce"
          style={{
            animationDelay: `${index * 0.1}s`,
            animationDuration: '0.6s',
          }}
        ></div>
      ))}
    </div>
  );
};

export default LoadingDots;