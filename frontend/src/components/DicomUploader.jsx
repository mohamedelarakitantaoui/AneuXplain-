import { useCallback, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
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
        } catch { /* fall back to status-code message */ }
        setError(msg);
      }
    };
    xhr.onerror = () => { setUploading(false); setError('Network error during upload'); };
    xhr.send(form);
  }, [apiUrl, onUploaded]);

  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation();
    setDragActive(false);
    const f = e.dataTransfer.files[0];
    if (f) doUpload(f);
  };

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); setDragActive(true); }}
      onDragLeave={(e) => { e.preventDefault(); e.stopPropagation(); setDragActive(false); }}
      style={{
        position: 'absolute', inset: 0,
        background: '#FAFBFF',
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        paddingLeft: 90, paddingRight: 24,
      }}
    >
      {/* Subtle grid texture */}
      <div style={{
        position: 'absolute', inset: 0, pointerEvents: 'none',
        backgroundImage: 'radial-gradient(circle, rgba(220,38,38,0.04) 1px, transparent 1px)',
        backgroundSize: '28px 28px',
        zIndex: 0,
      }} />

      <input
        ref={inputRef}
        type="file"
        accept=".nii,.nii.gz,.zip"
        onChange={(e) => e.target.files[0] && doUpload(e.target.files[0])}
        style={{ display: 'none' }}
      />

      <motion.div
        initial={{ opacity: 0, y: 22 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        style={{ width: '100%', maxWidth: 520, position: 'relative', zIndex: 1 }}
      >
        {/* Branding header */}
        <div style={{ textAlign: 'center', marginBottom: 40 }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
            <img src="/logoAneuX.png" alt="AneuXplain" style={{ width: 36, height: 36, objectFit: 'contain' }} />
            <span style={{ fontSize: 18, fontWeight: 700, color: '#000229', letterSpacing: '-0.02em', fontFamily: 'Syne, sans-serif' }}>
              AneuXplain
            </span>
          </div>
          <div style={{ fontSize: 28, fontWeight: 800, color: '#000229', letterSpacing: '-0.03em', lineHeight: 1.15, fontFamily: 'Syne, sans-serif', marginBottom: 10 }}>
            Upload DICOM / NIfTI Scan
          </div>
          <div style={{ fontSize: 14, color: '#64748B', lineHeight: 1.6, maxWidth: 380, margin: '0 auto' }}>
            Upload a neuroimaging scan to extract vessel geometry and perform AI-based aneurysm rupture risk assessment.
          </div>
        </div>

        {/* Drop zone card */}
        <motion.div
          animate={{
            borderColor: dragActive ? '#dc2626' : 'rgba(220,38,38,0.2)',
            background: dragActive ? 'rgba(220,38,38,0.03)' : '#ffffff',
            boxShadow: dragActive
              ? '0 0 0 4px rgba(220,38,38,0.08), 0 8px 32px rgba(220,38,38,0.06)'
              : '0 2px 16px rgba(0,0,0,0.05), 0 0 0 1px rgba(0,0,0,0.04)',
          }}
          transition={{ duration: 0.18 }}
          onClick={() => !uploading && inputRef.current?.click()}
          style={{
            border: '1.5px dashed rgba(220,38,38,0.2)',
            borderRadius: 24,
            padding: '48px 40px 44px',
            textAlign: 'center',
            cursor: uploading ? 'default' : 'pointer',
            marginBottom: 20,
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          {/* Corner accents */}
          <div style={{ position: 'absolute', top: 14, left: 14, width: 18, height: 18, borderTop: '2px solid #dc2626', borderLeft: '2px solid #dc2626', borderRadius: '4px 0 0 0', opacity: 0.4 }} />
          <div style={{ position: 'absolute', top: 14, right: 14, width: 18, height: 18, borderTop: '2px solid #dc2626', borderRight: '2px solid #dc2626', borderRadius: '0 4px 0 0', opacity: 0.4 }} />
          <div style={{ position: 'absolute', bottom: 14, left: 14, width: 18, height: 18, borderBottom: '2px solid #dc2626', borderLeft: '2px solid #dc2626', borderRadius: '0 0 0 4px', opacity: 0.4 }} />
          <div style={{ position: 'absolute', bottom: 14, right: 14, width: 18, height: 18, borderBottom: '2px solid #dc2626', borderRight: '2px solid #dc2626', borderRadius: '0 0 4px 0', opacity: 0.4 }} />

          {/* Upload icon */}
          <motion.div
            animate={{ scale: dragActive ? 1.1 : 1 }}
            transition={{ type: 'spring', stiffness: 300, damping: 20 }}
            style={{
              width: 72, height: 72, borderRadius: '50%',
              background: dragActive ? 'rgba(220,38,38,0.12)' : 'rgba(220,38,38,0.06)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              margin: '0 auto 24px',
              border: '1px solid rgba(220,38,38,0.15)',
            }}
          >
            {uploading
              ? <Loader2 style={{ width: 28, height: 28, color: '#dc2626', animation: 'spin 1s linear infinite' }} />
              : <Upload style={{ width: 28, height: 28, color: '#dc2626' }} />
            }
          </motion.div>

          <div style={{ fontSize: 20, fontWeight: 700, color: '#000229', marginBottom: 8, letterSpacing: '-0.02em', fontFamily: 'Syne, sans-serif' }}>
            {uploading
              ? `Uploading… ${progress}%`
              : dragActive
                ? 'Release to upload'
                : 'Drop your scan file here'
            }
          </div>
          <div style={{ fontSize: 13, color: '#94A3B8', lineHeight: 1.6 }}>
            {uploading ? 'Please wait while your file is being processed' : 'or click anywhere to browse your files'}
          </div>

          {/* Progress bar */}
          {uploading && (
            <div style={{ marginTop: 24, height: 3, background: 'rgba(220,38,38,0.12)', borderRadius: 2, overflow: 'hidden', maxWidth: 260, margin: '24px auto 0' }}>
              <motion.div
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.2 }}
                style={{ height: '100%', background: '#dc2626', borderRadius: 2 }}
              />
            </div>
          )}
        </motion.div>

        {/* Format badges + button row */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {['.nii', '.nii.gz', '.zip'].map((ext) => (
              <span key={ext} style={{
                fontSize: 11, fontWeight: 600, color: '#dc2626',
                background: 'rgba(220,38,38,0.06)', borderRadius: 6,
                padding: '4px 10px', fontFamily: 'ui-monospace, monospace',
                border: '1px solid rgba(220,38,38,0.15)',
                letterSpacing: '0.02em',
              }}>
                {ext}
              </span>
            ))}
          </div>
          <button
            onClick={() => inputRef.current?.click()}
            disabled={uploading}
            style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '10px 24px',
              background: uploading ? '#E2E8F0' : '#dc2626',
              color: uploading ? '#94A3B8' : '#ffffff',
              fontWeight: 600, fontSize: 13,
              borderRadius: 10, border: 'none',
              cursor: uploading ? 'not-allowed' : 'pointer',
              boxShadow: uploading ? 'none' : '0 4px 14px rgba(220,38,38,0.3)',
              whiteSpace: 'nowrap',
              flexShrink: 0,
              fontFamily: 'Syne, sans-serif',
            }}
          >
            <Upload style={{ width: 15, height: 15 }} />
            Select File
          </button>
        </div>

        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                marginTop: 20,
                background: 'rgba(239,68,68,0.06)',
                border: '1px solid rgba(239,68,68,0.2)',
                borderRadius: 10, padding: '9px 14px',
              }}
            >
              <AlertTriangle style={{ width: 14, height: 14, color: '#ef4444', flexShrink: 0 }} />
              <span style={{ color: '#ef4444', fontSize: 12 }}>{error}</span>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}
