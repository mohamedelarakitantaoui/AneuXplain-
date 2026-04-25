import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Crosshair, Loader2, AlertTriangle } from 'lucide-react';
import DicomUploader from './DicomUploader';
import DicomSliceBrowser from './DicomSliceBrowser';
import VesselTreeViewer from './VesselTreeViewer';

const TRANSITION = { duration: 0.3, ease: [0.22, 1, 0.36, 1] };

export default function DicomWorkflow({ apiUrl, onAnalysisComplete, initialSessionId = null }) {
  // If re-entering with an existing session (user clicked "Back to DICOM"),
  // skip the upload + segmentation screens and go straight to the vessel
  // tree — segmentation output is still cached in the backend session.
  const [sessionId, setSessionId] = useState(initialSessionId);
  const [metadata, setMetadata] = useState(null);
  const [sliceInfo, setSliceInfo] = useState(null);
  const [segmenting, setSegmenting] = useState(false);
  const [segmentationDone, setSegmentationDone] = useState(Boolean(initialSessionId));
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState(null);

  // Fetch slice info once upload is done
  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    fetch(`${apiUrl}/dicom/slice-info/${sessionId}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('slice-info failed'))))
      .then((data) => { if (!cancelled) setSliceInfo(data); })
      .catch((e) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [apiUrl, sessionId]);

  const handleUploaded = (sid, meta) => {
    setSessionId(sid);
    setMetadata(meta);
    setSegmentationDone(false);
    setSliceInfo(null);
    setError(null);
  };

  const handleSegment = async () => {
    if (!sessionId) return;
    setSegmenting(true);
    setError(null);
    try {
      const res = await fetch(`${apiUrl}/dicom/segment/${sessionId}?threshold_percentile=99.0`, {
        method: 'POST',
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `Segmentation failed (${res.status})`);
      }
      setSegmentationDone(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setSegmenting(false);
    }
  };

  const handleCropAndAnalyze = async (clickPointMm, cropRadiusMm) => {
    if (!sessionId) return;
    setAnalyzing(true);
    setError(null);
    try {
      const res = await fetch(`${apiUrl}/dicom/crop-and-analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          click_point: clickPointMm,
          crop_radius_mm: cropRadiusMm,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `Analyze failed (${res.status})`);
      }
      const payload = await res.json();
      // Cache-bust so re-clicking on a different region forces ArteryViewer
      // to re-load the freshly-written cropped .obj instead of the previous one.
      const croppedMeshUrl = `${apiUrl}/dicom/cropped-mesh/${sessionId}.obj?t=${Date.now()}`;
      onAnalysisComplete(payload, { croppedMeshUrl, sessionId });
    } catch (e) {
      setError(e.message);
    } finally {
      setAnalyzing(false);
    }
  };

  // ===== Render =====
  if (!sessionId) {
    return <DicomUploader apiUrl={apiUrl} onUploaded={handleUploaded} />;
  }

  return (
    <div className="absolute inset-0 flex flex-col" style={{ background: '#ffffff' }}>
      {/* Header strip */}
      <div
        className="flex items-center justify-between px-5"
        style={{ height: 56, borderBottom: '1px solid rgba(0,0,0,0.06)' }}
      >
        <div style={{ fontSize: 12, color: '#64748B', fontFamily: 'ui-monospace, monospace' }}>
          session <span style={{ color: '#dc2626' }}>{sessionId.slice(0, 8)}</span>
          {metadata?.source && <span style={{ marginLeft: 12, color: '#475569' }}>• {metadata.source}</span>}
        </div>
        <div style={{ fontSize: 11, color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
          {segmentationDone ? 'Step 2 — Click to crop' : 'Step 1 — Browse & segment'}
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 relative" style={{ minHeight: 0 }}>
        <AnimatePresence mode="wait">
          {!segmentationDone ? (
            <motion.div
              key="browse"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={TRANSITION}
              className="absolute inset-0 flex flex-col items-center"
              style={{ padding: 24 }}
            >
              <div style={{ flex: 1, width: '100%', maxWidth: 720, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
                {sliceInfo ? (
                  <DicomSliceBrowser apiUrl={apiUrl} sessionId={sessionId} axisCounts={sliceInfo} />
                ) : (
                  <div className="flex-1 flex items-center justify-center">
                    <Loader2 style={{ width: 24, height: 24, color: '#dc2626', animation: 'spin 1s linear infinite' }} />
                  </div>
                )}
              </div>

              <button
                onClick={handleSegment}
                disabled={segmenting || !sliceInfo}
                className="flex items-center gap-2.5 mt-5"
                style={{
                  padding: '12px 28px',
                  background: segmenting || !sliceInfo ? 'rgba(0,0,0,0.05)' : 'linear-gradient(135deg, #dc2626, #dc2626)',
                  color: segmenting || !sliceInfo ? '#94A3B8' : '#fff',
                  fontWeight: 500,
                  fontSize: 14,
                  borderRadius: 10,
                  border: 'none',
                  cursor: segmenting || !sliceInfo ? 'not-allowed' : 'pointer',
                  boxShadow: segmenting || !sliceInfo ? 'none' : '0 4px 16px rgba(74, 158, 255, 0.25)',
                }}
              >
                {segmenting ? (
                  <Loader2 style={{ width: 16, height: 16, animation: 'spin 1s linear infinite' }} />
                ) : (
                  <Activity style={{ width: 16, height: 16 }} />
                )}
                {segmenting ? 'Segmenting...' : 'Run Segmentation'}
              </button>
            </motion.div>
          ) : (
            <motion.div
              key="vessel"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={TRANSITION}
              className="absolute inset-0"
            >
              <VesselTreeViewer
                apiUrl={apiUrl}
                sessionId={sessionId}
                onCropPointSelected={handleCropAndAnalyze}
              />

              {/* Analyzing overlay */}
              <AnimatePresence>
                {analyzing && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 flex items-center justify-center pointer-events-none"
                    style={{ background: 'rgba(15, 17, 23, 0.88)', backdropFilter: 'blur(4px)', zIndex: 50 }}
                  >
                    <div className="text-center">
                      <Loader2 style={{ width: 40, height: 40, color: '#dc2626', animation: 'spin 1s linear infinite', margin: '0 auto 16px' }} />
                      <p style={{ color: '#64748B', fontSize: 12, fontWeight: 400, textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                        Cropping & analyzing region...
                      </p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Help hint when nothing clicked yet */}
              <div
                className="absolute top-3 right-3 flex items-center gap-2"
                style={{
                  background: 'rgba(26, 29, 39, 0.9)',
                  backdropFilter: 'blur(24px)',
                  border: '1px solid rgba(255,255,255,0.06)',
                  borderRadius: 20,
                  padding: '6px 12px',
                  fontSize: 11,
                  color: '#64748B',
                }}
              >
                <Crosshair style={{ width: 12, height: 12 }} />
                Click the vessel tree to select a crop center
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error toast */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 50 }}
              className="absolute bottom-20 left-1/2 -translate-x-1/2 flex items-center gap-2.5"
              style={{
                background: 'rgba(239, 68, 68, 0.9)',
                backdropFilter: 'blur(12px)',
                borderRadius: 10,
                padding: '8px 16px',
                zIndex: 60,
              }}
            >
              <AlertTriangle style={{ width: 14, height: 14, color: '#fff' }} />
              <span style={{ color: '#fff', fontSize: 13 }}>{error}</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
