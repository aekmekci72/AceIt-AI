import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from './AuthContext';
import Logo from './assets/logo (2).png';

const Navbar = () => {
    const navigate = useNavigate();
    const [searchTerm, setSearchTerm] = useState('');
    const [searchCategory, setSearchCategory] = useState('');
    const { logout } = useAuth();
    const [isOpen, setIsOpen] = useState(false);

    const handleKeyPress = (event) => {
        if (event.key === 'Enter' && searchCategory) {
            localStorage.setItem('searchTerm', searchTerm);
            localStorage.setItem('searchCategory', searchCategory);
            navigate(`/search`);
        }
    };

    const handleLogout = () => {
        logout();
        navigate('/');
    };

    return (
        <nav className="bg-gray-800 shadow-lg">
            <div className="w-full mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                    <div className="flex items-center flex-shrink-0">
                        <Link to="/homepage" className="flex items-center">
                            <img className="h-8 w-auto mr-2" src={Logo} alt="Logo" />
                            <span className="text-white text-lg font-semibold hidden sm:block">AceIt AI</span>
                        </Link>
                    </div>
                    <div className="flex-1 flex items-center justify-center">
                        <div className="w-full max-w-lg">
                            <div className="relative flex items-center">
                                <select
                                    className="w-32 px-3 py-2 bg-gray-700 text-gray-300 rounded-l-md border border-transparent focus:outline-none focus:bg-gray-600 focus:text-gray-100"
                                    value={searchCategory}
                                    onChange={(e) => setSearchCategory(e.target.value)}
                                >
                                    <option value="" disabled>Select...</option>
                                    <option value="publicsets">Public Sets</option>
                                    <option value="users">Users</option>
                                </select>
                                <input
                                    id="search"
                                    name="search"
                                    className="block flex-grow pl-2 pr-3 py-2 border border-transparent rounded-r-md leading-5 bg-gray-700 text-gray-300 placeholder-gray-400 focus:outline-none focus:bg-gray-600 focus:text-gray-100 sm:text-sm"
                                    placeholder="Search..."
                                    type="search"
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                    onKeyPress={handleKeyPress}
                                    disabled={!searchCategory}
                                />
                            </div>
                        </div>
                    </div>
                    <div className="hidden md:flex items-center space-x-2 flex-shrink-0">
                        <NavButton onClick={() => navigate('/mygallery')}>Your Collections</NavButton>
                        <NavButton onClick={() => navigate('/studytimelines')}>Study Timelines</NavButton>
                        <NavButton onClick={() => navigate('/followers')}>Followers</NavButton>
                        <NavButton onClick={handleLogout} className="bg-cyan-600 hover:bg-cyan-700">Logout</NavButton>
                    </div>
                    <div className="md:hidden relative flex-shrink-0">
                        <button
                            onClick={() => setIsOpen(!isOpen)}
                            className="text-gray-300 hover:bg-gray-700 hover:text-white px-3 py-2 rounded-md text-sm font-medium"
                        >
                            <svg className="h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                            </svg>
                        </button>
                        {isOpen && (
                            <div className="absolute right-0 mt-2 w-48 bg-gray-800 rounded-md overflow-hidden shadow-xl z-10">
                                <NavButton onClick={() => navigate('/mygallery')} className="w-full text-left">Your Collections</NavButton>
                                <NavButton onClick={() => navigate('/studytimelines')} className="w-full text-left">Study Timelines</NavButton>
                                <NavButton onClick={() => navigate('/followers')} className="w-full text-left">Followers</NavButton>
                                <NavButton onClick={handleLogout} className="w-full text-left bg-cyan-600 hover:bg-cyan-700">Logout</NavButton>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </nav>
    );
};

const NavButton = ({ onClick, children, className = "" }) => (
    <button
        onClick={onClick}
        className={`whitespace-nowrap text-white px-3 py-2 rounded-md text-sm font-medium transition duration-300 ease-in-out transform hover:scale-105 ${className} ${className ? '' : 'bg-gray-700 hover:bg-gray-600'}`}
    >
        {children}
    </button>
);

export default Navbar;