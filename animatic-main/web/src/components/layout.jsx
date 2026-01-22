import { Outlet, Link, useLocation } from "react-router-dom";
import {Zap,} from 'lucide-react';
import { Toaster } from "react-hot-toast";

const Layout = () => {
    const location = useLocation()
    
    const currentPath = location.pathname;

    const currentTab =
        currentPath === "/" ? "presets" :
        currentPath === "/recent-files" ? "recent" : "";
    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-900 to-slate-900">
            <div className="bg-black/20 backdrop-blur-sm border-b border-slate-500/20">
                <div className="max-w-7xl mx-auto px-4 py-6">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 bg-gradient-to-r from-slate-500 to-pink-500 rounded-lg flex items-center justify-center">
                            <Zap className="w-6 h-6 text-white" />
                        </div>
                        <div>
                            <p className="text-sm md:text-5xl font-bold text-white">Animatic</p>
                            <p className="hidden md:block text-slate-300 text-sm">2D to 3D Conversion Studio</p>
                        </div>
                        </div>
                        
                        <div className="flex flex-col sm:flex-row sm:space-x-1 space-y-1 sm:space-y-0 bg-black/30 rounded-lg p-1">
                        <Link
                            to="/"
                            style={{
                                color:"white"
                            }}
                            className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                            currentTab === 'presets'
                                ? 'bg-slate-600 text-white shadow-lg'
                                : 'text-slate-300 hover:text-white'
                            }`}
                        >
                            Create New
                        </Link>

                        <Link
                            to="/recent-files"
                            style={{
                                color:"white"
                            }}
                            className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                            currentTab === 'recent'
                                ? 'bg-slate-600 text-white shadow-lg'
                                : 'text-slate-300 hover:text-white'
                            }`}
                        >
                            Recent Files
                        </Link>
                        </div>
                    </div>
                </div>
            </div>
            <Toaster position="top-right" />
            <Outlet />
        </div>
    )
}

export default Layout;