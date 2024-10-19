import React from 'react';
import '../styles/components/button.css';

const Button = ({ text, destination, color, textcolor }) => {
  const handleClick = () => {
    window.location.href = destination;
  };

  const buttonStyle = {
    backgroundColor: color,
    color: textcolor
  };

  return (
    <button className="category-btn" style={buttonStyle} onClick={handleClick}>
      {text}
    </button>
  );
};

export default Button;
