import { useCallback, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Upload, Loader2, AlertTriangle } from 'lucide-react';

const ACCEPTED = ['.nii', '.nii.gz', '.zip'];

function isAccepted(name) {
  const lower = name.toLowerCase();
  return ACCEPTED.some((ext) => lower.endsWith(ext));
}

export default function DicomUploader({ apiUrl, onUploaded }) {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const doUpload = useCallback((file) => {
    if (!isAccepted(file.name)) {
      setError('Please upload a .nii, .nii.gz, or .zip file');
      return;
    }
    setError(null);
    setUploading(true);
    setProgress(0);

    const form = new FormData();
    form.append('file', file);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${apiUrl}/dicom/upload`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () => {
      setUploading(false);
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText);
          onUploaded(data.session_id, data.metadata);
        } catch {
          setError('Invalid server response');
        }
      } else {
        let msg = `Upload failed (${xhr.status})`;
        try {
          const data = JSON.parse(xhr.responseText);
          if (data.detail) msg = data.detail;
        } catch {
          // ignore — fall back to status-code message
        }
        setError(msg);
      }
    };
    xhr.onerror = () => {
      setUploading(false);
      setError('Network error during upload');
    };
    xhr.send(form);
  }, [apiUrl, onUploaded]);

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const f = e.dataTransfer.files[0];
    if (f) doUpload(f);
  };

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); setDragActive(true); }}
      onDragLeave={(e) => { e.preventDefault(); e.stopPropagation(); setDragActive(false); }}
      className={`absolute inset-0 flex flex-col items-center justify-center transition-all duration-300 ${
        dragActive ? 'ring-2 ring-inset ring-[#4A9EFF]' : ''
      }`}
      style={{ background: 'linear-gradient(135deg, #0F1117 0%, #141821 50%, #0F1117 100%)' }}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".nii,.nii.gz,.zip"
        onChange={(e) => e.target.files[0] && doUpload(e.target.files[0])}
        className="hidden"
      />
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <div
          className="w-28 h-28 mx-auto mb-8 flex items-center justify-center"
          style={{ background: 'rgba(26, 29, 39, 0.6)', borderRadius: 20, border: '1px dashed rgba(255,255,255,0.08)' }}
        >
          {uploading ? (
            <Loader2 style={{ width: 48, height: 48, color: '#4A9EFF', animation: 'spin 1s linear infinite' }} />
          ) : (
            <Upload style={{ width: 48, height: 48, color: '#374151' }} />
          )}
        </div>
        <h3 style={{ fontSize: 22, fontWeight: 300, color: '#F1F5F9', marginBottom: 8, letterSpacing: '-0.01em' }}>
          Drop DICOM or NIfTI scan
        </h3>
        <p style={{ color: '#64748B', marginBottom: 36, maxWidth: 420, fontSize: 13, fontWeight: 400, lineHeight: 1.6 }}>
          Upload a .nii / .nii.gz volume or a zipped DICOM series — drag and drop or click to browse
        </p>
        <button
          onClick={() => inputRef.current?.click()}
          disabled={uploading}
          className="flex items-center gap-2.5 mx-auto"
          style={{
            padding: '12px 28px',
            background: 'linear-gradient(135deg, #4A9EFF, #3B82F6)',
            color: '#fff',
            fontWeight: 500,
            fontSize: 14,
            borderRadius: 10,
            border: 'none',
            cursor: uploading ? 'not-allowed' : 'pointer',
            opacity: uploading ? 0.6 : 1,
            boxShadow: '0 4px 16px rgba(74, 158, 255, 0.25)',
          }}
        >
          <Upload style={{ width: 18, height: 18 }} />
          {uploading ? `Uploading ${progress}%` : 'Select File'}
        </button>

        {uploading && (
          <div style={{ marginTop: 18, width: 260, marginLeft: 'auto', marginRight: 'auto' }}>
            <div style={{ height: 3, background: 'rgba(255,255,255,0.06)', borderRadius: 2, overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${progress}%`, background: '#4A9EFF', transition: 'width 0.2s ease' }} />
            </div>
          </div>
        )}

        {error && (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center gap-2 mx-auto mt-6"
            style={{
              background: 'rgba(239, 68, 68, 0.12)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: 10,
              padding: '8px 14px',
              maxWidth: 360,
            }}
          >
            <AlertTriangle style={{ width: 14, height: 14, color: '#ef4444' }} />
            <span style={{ color: '#fca5a5', fontSize: 12 }}>{error}</span>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}
