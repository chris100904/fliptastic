import React, { useState, useRef } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { FaEyeSlash, FaEye, FaUser } from 'react-icons/fa';
import './App.css';
import './index.css';

// configure pdf.js worker
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url
).toString();
function App() {
  const [mode, setMode] = useState('home');
  return (
    <div className="App">
      {mode === 'home' ? (
        <HomePage onChooseMode={setMode} />
      ) : (
        <ModePage mode={mode} onBack={() => setMode('home')} />
      )}
    </div>
  );
}

function HomePage({ onChooseMode }) {
  return (
    <>
      <header className="header">
        <h1 className="logo">
          {'Fliptastic'.split('').map((ch, i) => (
            <span key={i} className="logo-letter">{ch}</span>
          ))}
        </h1>
        <p className="subheading">
          Flip pages hands‑free: blink, gaze, or tilt!
        </p>
      </header>
      <div className="mode-grid">
        <ModeCard
          className="blink"
          icon={<FaEyeSlash />}
          title="Blink Mode"
          description="Blink three times in a row to flip."
          onSelect={() => onChooseMode('blink')}
        />
        <ModeCard
          className="gaze"
          icon={<FaEye />}
          title="Gaze Mode"
          description="Gaze left/right for 5+ seconds."
          onSelect={() => onChooseMode('gaze')}
        />
        <ModeCard
          className="head"
          icon={<FaUser />}
          title="Head Mode"
          description="Tilt head left/right for 3+ seconds."
          onSelect={() => onChooseMode('head')}
        />
      </div>
    </>
  );
}

function ModeCard({ icon, title, description, onSelect, className }) {
  return (
    <div className={`mode-card ${className}`} onClick={onSelect}>
      <div className="mode-icon">{icon}</div>
      <h2>{title}</h2>
      <p>{description}</p>
      <button className="mode-button">Select {title}</button>
    </div>
  );
}

function ModePage({ mode, onBack }) {
  const fileInputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [url, setUrl] = useState('');
  const [numPages, setNumPages] = useState(null);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const pickFile = (e) => {
    const f = e.target.files[0];
    setFile(f);
    setUrl('');
    setNumPages(null);
    setPage(1);
    setError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    pickFile({ target: { files: e.dataTransfer.files } });
  };

  const upload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append('pdf', file);
      const res = await fetch('http://localhost:8000/upload', { method: 'POST', body: form });
      const data = await res.json();
      setUrl(data.url);
    } catch {
      setError('Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const onLoadSuccess = ({ numPages }) => setNumPages(numPages);
  const next = () => setPage((p) => Math.min(p + 1, numPages));
  const prev = () => setPage((p) => Math.max(p - 1, 1));

  return (
    <div className="mode-page">
      <button className="back-button" onClick={onBack}>← Back</button>
      <h1 className="mode-title">{mode.charAt(0).toUpperCase() + mode.slice(1)} Mode</h1>
      <div
        className="drop-zone"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        onClick={() => fileInputRef.current.click()}
      >
        {file ? file.name : 'Drag & drop your PDF, or click to select'}
        <input
          ref={fileInputRef}
          type="file"
          accept="application/pdf"
          onChange={pickFile}
          style={{ display: 'none' }}
        />
      </div>
      <button
        className="upload-button"
        onClick={upload}
        disabled={!file || loading}
      >
        {loading ? 'Uploading…' : 'Upload & View'}
      </button>
      {error && <div className="error">{error}</div>}
      {url && (
        <div className="viewer-container">
          <div className="pager-controls">
            <button onClick={prev} disabled={page <= 1}>Prev</button>
            <span>{page} / {numPages}</span>
            <button onClick={next} disabled={page >= numPages}>Next</button>
          </div>
          <Document file={url} onLoadSuccess={onLoadSuccess} loading={<p>Loading…</p>}>
            <Page pageNumber={page} width={700} />
          </Document>
        </div>
      )}
    </div>
  );
}

export default App;
