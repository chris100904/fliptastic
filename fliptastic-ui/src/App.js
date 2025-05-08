import React, { useState } from "react";
import axios from "axios";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";

//pdfjs.GlobalWorkerOptions.workerSrc = `${process.env.PUBLIC_URL}/pdf.worker.min.js`;
//pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/4.8.69/pdf.worker.min.js`;

// pdfjs.GlobalWorkerOptions.workerSrc = `//cdn.jsdelivr.net/npm/pdfjs-dist@2.16.105`;

pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();

function App() {
  const [pdfFile, setPdfFile] = useState(null);
  const [pdfUrl, setPdfUrl] = useState("");
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const pickFile = (e) => {
    setPdfFile(e.target.files[0]);
    setPdfUrl("");
    setNumPages(null);
    setPageNumber(1);
    setError(null);
  };

  const upload = async () => {
    if (!pdfFile) return alert("Select a PDF first");

    setIsLoading(true);
    setError(null);

    try {
      const form = new FormData();
      form.append("pdf", pdfFile);
      const res = await axios.post("http://localhost:8000/upload", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setPdfUrl(res.data.url);
    } catch (err) {
      setError("Failed to upload PDF");
      console.error("Upload error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
    setError(null);
  };

  const onDocumentLoadError = (err) => {
    setError("Failed to load PDF document");
    console.error("Document load error:", err);
  };

  const onPageLoadError = (err) => {
    setError("Failed to load PDF page");
    console.error("Page load error:", err);
  };

  const nextPage = () => setPageNumber((p) => Math.min(p + 1, numPages));
  const prevPage = () => setPageNumber((p) => Math.max(p - 1, 1));

  return (
    <div style={{ padding: 20 }}>
      <h1>Fliptastic PDF Viewer</h1>

      <input type="file" accept="application/pdf" onChange={pickFile} disabled={isLoading} />
      <button onClick={upload} style={{ marginLeft: 10 }} disabled={isLoading || !pdfFile}>
        {isLoading ? "Uploading..." : "Upload & View"}
      </button>

      {error && <div style={{ color: "red", margin: "10px 0" }}>Error: {error}</div>}

      {pdfUrl && (
        <>
          <p>PDF URL: {pdfUrl}</p>
          <a href={pdfUrl} target="_blank" rel="noopener noreferrer" style={{ display: "block", marginBottom: 10 }}>
            ▶️ Open PDF in new tab
          </a>

          <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
            <div>
              <button onClick={prevPage} disabled={pageNumber <= 1 || isLoading}>
                Previous
              </button>
              <span style={{ margin: "0 10px" }}>
                Page {pageNumber} of {numPages || "?"}
              </span>
              <button onClick={nextPage} disabled={pageNumber >= (numPages || 1) || isLoading}>
                Next
              </button>
            </div>

            <Document
              file={pdfUrl}
              onLoadSuccess={onDocumentLoadSuccess}
              onLoadError={onDocumentLoadError}
              loading={<div>Loading PDF...</div>}
            >
              <Page
                pageNumber={pageNumber}
                width={800}
                onLoadError={onPageLoadError}
                loading={<div>Loading page {pageNumber}...</div>}
              />
            </Document>
          </div>
        </>
      )}
    </div>
  );
}

export default App;
