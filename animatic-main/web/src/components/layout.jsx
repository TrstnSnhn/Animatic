import { Outlet, Link, useLocation } from "react-router-dom";
import { Toaster } from "react-hot-toast";

const Layout = () => {
  const location = useLocation();
  const currentPath = location.pathname;

  const currentTab =
    currentPath === "/" ? "presets" : currentPath === "/recent-files" ? "recent" : "";

  return (
    <div className="min-h-screen bg-neutral-950 text-white overflow-x-hidden">
      {/* Background effects - full width */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-anim-vignette" />
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[200%] h-[60%] bg-anim-glow blur-3xl opacity-60" />
      </div>

      {/* Content */}
      <div className="relative min-h-screen flex flex-col">
        <header className="w-full px-6 pt-10">
          <div className="flex flex-col items-center gap-6">
            <div className="flex items-center gap-3">
              {/* Custom Logo */}
              <img 
                src="/logo.png" 
                alt="Animatic Logo" 
                className="h-11 w-11 object-contain"
              />

              <div className="leading-tight text-left">
                <div className="text-2xl font-semibold tracking-tight">Animatic</div>
                <div className="text-xs text-white/60">2D to 3D Conversion Studio</div>
              </div>
            </div>

            <div className="glass-soft flex items-center gap-1 p-1">
              <Link to="/" className={currentTab === "presets" ? "tab-active" : "tab"}>
                Create New
              </Link>
              <Link
                to="/recent-files"
                className={currentTab === "recent" ? "tab-active" : "tab"}
              >
                Recent Files
              </Link>
            </div>
          </div>
        </header>

        <Toaster position="top-right" />

        <main className="flex-1 w-full max-w-5xl mx-auto px-6 pb-16 pt-10">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default Layout;