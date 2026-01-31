import { Outlet, Link, useLocation } from "react-router-dom";
import { Zap } from "lucide-react";
import { Toaster } from "react-hot-toast";

const Layout = () => {
  const location = useLocation();
  const currentPath = location.pathname;

  const currentTab =
    currentPath === "/" ? "presets" : currentPath === "/recent-files" ? "recent" : "";

  return (
    <div className="min-h-screen bg-ink-950 text-white">
      <div className="relative min-h-screen overflow-hidden">
        <div className="pointer-events-none absolute inset-0 bg-anim-vignette" />
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-[55%] bg-anim-glow blur-3xl opacity-70" />

        <header className="relative mx-auto w-full max-w-6xl px-6 pt-10">
          <div className="flex flex-col items-center gap-6">
            <div className="flex items-center gap-3">
              <div className="h-11 w-11 rounded-2xl bg-anim-badge p-[2px]">
                <div className="flex h-full w-full items-center justify-center rounded-[1.05rem] bg-ink-950">
                  <Zap className="h-5 w-5 text-white" />
                </div>
              </div>

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

        <main className="relative mx-auto w-full max-w-6xl px-6 pb-16 pt-10">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default Layout;
