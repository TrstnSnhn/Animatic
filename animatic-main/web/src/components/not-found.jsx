import React, { useState, useEffect } from 'react';
import { Home, ArrowLeft, Zap, Search, FileX, Sparkles } from 'lucide-react';
import { Link } from 'react-router-dom';

const NotFound = () => {
    const [glitchText, setGlitchText] = useState('404');
    const [floatingElements, setFloatingElements] = useState([]);

    useEffect(() => {
        const elements = Array.from({ length: 15 }, (_, i) => ({
        id: i,
        x: Math.random() * 100,
        y: Math.random() * 100,
        size: Math.random() * 20 + 10,
        delay: Math.random() * 5
        }));
        setFloatingElements(elements);

        const glitchInterval = setInterval(() => {
        const glitchChars = ['4', '0', '4', '█', '▓', '▒', '░'];
        const glitched = Array.from('404').map(() => 
            Math.random() < 0.1 ? glitchChars[Math.floor(Math.random() * glitchChars.length)] : '404'[Math.floor(Math.random() * 3)]
        ).join('');
        setGlitchText(glitched);
        
        setTimeout(() => setGlitchText('404'), 100);
        }, 2000);

        return () => clearInterval(glitchInterval);
    }, []);

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-900 to-slate-900 relative overflow-hidden">
        {floatingElements.map((element) => (
            <div
            key={element.id}
            className="absolute opacity-10 pointer-events-none"
            style={{
                left: `${element.x}%`,
                top: `${element.y}%`,
                width: `${element.size}px`,
                height: `${element.size}px`,
                animationDelay: `${element.delay}s`
            }}
            >
            <div className="w-full h-full bg-gradient-to-r from-slate-500 to-pink-500 rounded-full animate-pulse"></div>
            </div>
        ))}

        <div className="flex items-center justify-center min-h-[calc(100vh-80px)] px-4 relative z-10">
            <div className="text-center max-w-2xl mx-auto">
            <div className="mb-8 relative">
                <div className="text-6xl md:text-[12rem] font-black text-transparent bg-gradient-to-r from-slate-500 via-pink-500 to-slate-600 bg-clip-text leading-none">
                {glitchText}
                </div>
                <div className="absolute inset-0 text-6xl md:text-[12rem] font-black text-slate-500/20 blur-sm">
                404
                </div>
                
                <div className="absolute top-4 left-1/4 animate-bounce">
                <Sparkles className="w-6 h-6 text-slate-400" style={{ animationDelay: '0.5s' }} />
                </div>
                <div className="absolute top-1/3 right-1/4 animate-bounce">
                <Sparkles className="w-4 h-4 text-pink-400" style={{ animationDelay: '1s' }} />
                </div>
                <div className="absolute bottom-1/4 left-1/3 animate-bounce">
                <Sparkles className="w-5 h-5 text-slate-300" style={{ animationDelay: '1.5s' }} />
                </div>
            </div>

            <div className="bg-black/40 backdrop-blur-sm rounded-2xl border border-slate-500/30 p-8 mb-8">
                
                <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
                Oops! Page Not Found
                </h2>
                
                <p className="text-slate-300 text-lg mb-6 leading-relaxed">
                Looks like this page got lost in the 3D conversion process. The page you're looking for doesn't exist or may have been moved to another dimension.
                </p>

                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Link
                    to="/"
                    style={{ color: "white"}}
                    className="group flex items-center justify-center space-x-3 bg-gradient-to-r from-slate-600 to-pink-600 hover:from-slate-700 hover:to-pink-700 text-white py-4 px-8 rounded-xl font-semibold text-lg transition-all duration-300 transform hover:scale-105 shadow-xl"
                >
                    <Home className="w-5 h-5 group-hover:scale-110 transition-transform" />
                    <span>Go Home</span>
                </Link>
                
                </div>
            </div>

            <div className="text-center">
                <div className="inline-flex items-center space-x-2 text-slate-400 animate-pulse">
                <div className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: '0s' }}></div>
                <div className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                <div className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                </div>
                <p className="text-slate-400 text-sm mt-2">Lost in the digital void...</p>
            </div>
            </div>
        </div>

        <div className="absolute inset-0 opacity-5">
            <div className="absolute inset-0" style={{
            backgroundImage: `
                linear-gradient(rgba(147, 51, 234, 0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(147, 51, 234, 0.1) 1px, transparent 1px)
            `,
            backgroundSize: '50px 50px',
            animation: 'gridMove 20s linear infinite'
            }}></div>
        </div>

        <style jsx>{`
            @keyframes gridMove {
            0% { transform: translate(0, 0); }
            100% { transform: translate(50px, 50px); }
            }
        `}</style>
        </div>
    );
}

export default NotFound