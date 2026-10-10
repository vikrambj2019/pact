import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import WorkspacePage from './pages/WorkspacePage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/w/:workspaceId/*" element={<WorkspacePage />} />
      </Routes>
    </BrowserRouter>
  );
}
