import './app.css'
import { BrowserRouter, Routes, Route } from "react-router-dom";
import RecentFiles from './components/recent-files';
import Home from './components/home';
import Layout from './components/layout';
import NotFound from './components/not-found';

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="recent-files" element={<RecentFiles />} />
          {/* <Route path="contact" element={<Contact />} /> */}
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
