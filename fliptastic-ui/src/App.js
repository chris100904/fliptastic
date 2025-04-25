import React, { useState } from "react";
import axios from "axios";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";

// Set workerSrc for react-pdf
pdfjs.GlobalWorkerOptions.workerSrc = `${process.env.PUBLIC_URL}/pdf.worker.min.js`;

function App() {
  const [pdfFile, setPdfFile] = useState(null);
  const [pdfUrl, setPdfUrl] = useState("");
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);

  const pickFile = e => {
    setPdfFile(e.target.files[0]);
    setPdfUrl("");
    setNumPages(null);
    setPageNumber(1);
  };

  const upload = async () => {
    if (!pdfFile) return alert("Select a PDF first");
    const form = new FormData();
    form.append("pdf", pdfFile);
    const res = await axios.post("http://localhost:8000/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    setPdfUrl(res.data.url);
  };

  const onDocumentLoadSuccess = ({ numPages }) => setNumPages(numPages);

  const nextPage = () => setPageNumber(p => Math.min(p + 1, numPages));
  const prevPage = () => setPageNumber(p => Math.max(p - 1, 1));

  return (
    <div style={{ padding: 20 }}>
      <h1>Fliptastic PDF Viewer</h1>

      <input type="file" accept="application/pdf" onChange={pickFile} />
      <button onClick={upload} style={{ marginLeft: 10 }}>
        Upload & View
      </button>

      {pdfUrl && (
        <>
          {/* sanity-check printout and manual test link */}
          <p>PDF URL: {pdfUrl}</p>
          <a
            href={pdfUrl}
            target="_blank"
            rel="noopener noreferrer"
            style={{ display: "block", marginBottom: 10 }}
          >
            ▶️ Open PDF in new tab
          </a>

          <Document
            file={pdfUrl}                            // make sure this really is your URL string
            onLoadSuccess={({ numPages }) => setNumPages(numPages)}
            onLoadError={(err) => console.error("Document load error:", err)}
          >
            <Page
              pageNumber={pageNumber}
              onLoadError={(err) => console.error("Page load error:", err)}
            />
          </Document>
        </>
      )}

    </div>
  );
}

export default App;