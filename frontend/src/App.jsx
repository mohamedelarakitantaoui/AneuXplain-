import { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  Loader2,
  X,
  AlertTriangle,
  CheckCircle,
} from 'lucide-react';
import ArteryViewer from './components/ArteryViewer';
import { PACSToolbar, MeasurementIndicator, ClippingIndicator } from './components/Overlay';
import RiskScoreCard from './components/RiskAnalysisHUD';
import MorphologyReport from './components/MorphologyReport';
import DicomWorkflow from './components/DicomWorkflow';
import LandingPage from './components/LandingPage';
import { exportAll } from './utils/generateReport';

const API_URL = 'http://localhost:8000';

// ============================================
// View mode segmented control (iOS-style)
// ============================================
function ViewModeControl({ viewMode, onViewModeChange, heatmapData, heatmapLoading }) {
  const modes = [
    { id: 'risk', label: 'Risk Score' },
    { id: 'heatmap', label: 'Heatmap', disabled: !heatmapData, loading: heatmapLoading },
    { id: 'measurements', label: 'Measurements' },
  ];

  return (
    <div style={{
      display: 'flex',
      background: 'rgba(255,255,255,0.92)',
      backdropFilter: 'blur(24px)',
      borderRadius: 24,
      padding: 3,
      border: '1px solid rgba(0,0,0,0.08)',
      boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
    }}>
      {modes.map((mode) => (
        <button
          key={mode.id}
          onClick={() => !mode.disabled && onViewModeChange(mode.id)}
          disabled={mode.disabled}
          style={{
            padding: '6px 14px',
            borderRadius: 20,
            fontSize: 12,
            fontWeight: 400,
            border: 'none',
            cursor: mode.disabled ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s ease',
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            background: viewMode === mode.id
              ? 'rgba(0,0,0,0.07)'
              : 'transparent',
            color: viewMode === mode.id
              ? '#0F1117'
              : mode.disabled
                ? '#CBD5E1'
                : '#64748B',
          }}
        >
          {mode.loading && !heatmapData && (
            <Loader2 style={{ width: 12, height: 12, animation: 'spin 1s linear infinite' }} />
          )}
          {mode.label}
        </button>
      ))}
    </div>
  );
}

// ============================================
// Wireframe toggle switch
// ============================================
function WireframeToggle({ wireframe, onChange }) {
  return (
    <button
      onClick={() => onChange(!wireframe)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '6px 12px',
        borderRadius: 20,
        border: '1px solid rgba(0,0,0,0.08)',
        background: wireframe ? 'rgba(220,38,38,0.08)' : 'rgba(255,255,255,0.92)',
        backdropFilter: 'blur(24px)',
        cursor: 'pointer',
        fontSize: 11,
        color: wireframe ? '#dc2626' : '#64748B',
        boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
        transition: 'all 0.2s ease',
      }}
    >
      <div style={{
        width: 24,
        height: 14,
        borderRadius: 7,
        background: wireframe ? '#dc2626' : '#CBD5E1',
        position: 'relative',
        transition: 'background 0.2s ease',
      }}>
        <div style={{
          width: 10,
          height: 10,
          borderRadius: '50%',
          background: '#fff',
          position: 'absolute',
          top: 2,
          left: wireframe ? 12 : 2,
          transition: 'left 0.2s ease',
        }} />
      </div>
      Wire
    </button>
  );
}


// ============================================
// MAIN APP COMPONENT
// ============================================
function App() {
  // Upload mode: 'mesh' (.obj/.ply/.stl) or 'dicom' (.nii/.nii.gz/.zip)
  const [uploadMode, setUploadMode] = useState('mesh');

  // Tracks whether the most recent analysis came via the DICOM flow,
  // so we only show the harmonization badge in that case.
  const [dicomAnalysis, setDicomAnalysis] = useState(null);

  // DICOM session id kept alive across analyses so "Back to DICOM" can
  // remount the workflow without re-uploading / re-segmenting.
  const [dicomSessionId, setDicomSessionId] = useState(null);

  // State
  const [file, setFile] = useState(null);
  const [originalObjUrl, setOriginalObjUrl] = useState(null);
  const [riskScore, setRiskScore] = useState(null);
  const [riskLevel, setRiskLevel] = useState(null);
  const [interpretation, setInterpretation] = useState(null);
  const [analysisStatus, setAnalysisStatus] = useState('idle');
  const [error, setError] = useState(null);
  const [wireframeMode, setWireframeMode] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  // Medical tools
  const [clippingY, setClippingY] = useState(2);
  const [isMeasuring, setIsMeasuring] = useState(false);
  const [currentMeasurement, setCurrentMeasurement] = useState(null);

  // Morphology + Clinical Report
  const [morphologyData, setMorphologyData] = useState(null);
  const [clinicalReport, setClinicalReport] = useState(null);

  // Heatmap
  const [heatmapData, setHeatmapData] = useState(null);
  const [heatmapLoading, setHeatmapLoading] = useState(false);

  // Measurement highlight + view mode
  const [activeHighlight, setActiveHighlight] = useState(null);
  const [viewMode, setViewMode] = useState('risk');

  // Camera reset key — increment to re-frame the mesh
  const [resetViewKey, setResetViewKey] = useState(0);

  // Export toast
  const [showExportToast, setShowExportToast] = useState(false);

  // Landing page
  const [showLanding, setShowLanding] = useState(false);

  const fileInputRef = useRef(null);
  const viewerRef = useRef(null);

  // ============================================
  // HANDLERS
  // ============================================

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.name.endsWith('.obj')) {
      handleUpload(droppedFile);
    } else {
      setError('Please upload a .obj file');
    }
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  }, []);

  const handleUpload = async (selectedFile) => {
    setFile(selectedFile);
    setError(null);
    setRiskScore(null);
    setRiskLevel(null);
    setInterpretation(null);
    setMorphologyData(null);
    setClinicalReport(null);
    setHeatmapData(null);
    setActiveHighlight(null);
    setAnalysisStatus('idle');
    setWireframeMode(false);
    setViewMode('risk');
    setHeatmapLoading(false);

    const objUrl = URL.createObjectURL(selectedFile);
    setOriginalObjUrl(objUrl);
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setError(null);
    setAnalysisStatus('loading');

    const formData = new FormData();
    formData.append('file', file);

    try {
      // POST /analyze
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis failed');
      }

      const data = await response.json();
      setRiskScore(data.risk_score);
      setRiskLevel(data.risk_level);
      setInterpretation(data.interpretation);
      setMorphologyData(data.morphology ?? null);
      setClinicalReport(data.clinical_report ?? null);
      setAnalysisStatus('success');

      // POST /heatmap (non-blocking)
      setHeatmapLoading(true);
      const heatmapForm = new FormData();
      heatmapForm.append('file', file);
      fetch(`${API_URL}/heatmap`, {
        method: 'POST',
        body: heatmapForm,
      })
        .then(res => res.ok ? res.json() : null)
        .then(hData => {
          if (hData?.heatmap) {
            setHeatmapData(hData.heatmap);
          }
          setHeatmapLoading(false);
        })
        .catch(err => {
          console.warn('Heatmap fetch failed (non-critical):', err);
          setHeatmapLoading(false);
        });

    } catch (err) {
      setError(err.message);
      setAnalysisStatus('error');
      console.error('Analysis error:', err);
    }
  };

  const handleParameterSelect = useCallback((parameterName, spatialData) => {
    console.log('[App] handleParameterSelect:', parameterName, spatialData);
    if (!parameterName) {
      setActiveHighlight(null);
      return;
    }
    setActiveHighlight({ parameterName, spatial: spatialData });
    setViewMode('measurements');
  }, []);

  const handleExportReport = () => {
    // Capture the 3D viewport screenshot
    const canvasImage = viewerRef.current?.captureCanvas?.() || null;

    exportAll({
      filename: file?.name || 'unknown',
      riskScore,
      riskLevel,
      morphologyData,
      clinicalReport,
      canvasImage,
    });

    // Show toast
    setShowExportToast(true);
    setTimeout(() => setShowExportToast(false), 2000);
  };

  const handleReset = () => {
    setFile(null);
    setOriginalObjUrl(null);
    setRiskScore(null);
    setRiskLevel(null);
    setInterpretation(null);
    setMorphologyData(null);
    setClinicalReport(null);
    setHeatmapData(null);
    setActiveHighlight(null);
    setAnalysisStatus('idle');
    setError(null);
    setWireframeMode(false);
    setViewMode('risk');
    setHeatmapLoading(false);
    setDicomAnalysis(null);
    setDicomSessionId(null);
  };

  // "Back to DICOM" — drop the current analysis view but keep the
  // session alive so DicomWorkflow can remount directly on the vessel
  // tree (segmentation is already cached server-side).
  const handleBackToDicom = () => {
    setFile(null);
    setOriginalObjUrl(null);
    setRiskScore(null);
    setRiskLevel(null);
    setInterpretation(null);
    setMorphologyData(null);
    setClinicalReport(null);
    setHeatmapData(null);
    setActiveHighlight(null);
    setAnalysisStatus('idle');
    setError(null);
    setWireframeMode(false);
    setViewMode('risk');
    setHeatmapLoading(false);
    setDicomAnalysis(null);
    setUploadMode('dicom');
    // dicomSessionId intentionally kept
  };

  // ============================================
  // RENDER
  // ============================================

  return (
    <div
      className="h-screen w-screen overflow-hidden flex"
      style={{ background: '#ffffff', color: '#0F1117', fontFamily: "'Inter', system-ui, sans-serif" }}
    >
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".obj"
        onChange={(e) => e.target.files[0] && handleUpload(e.target.files[0])}
        className="hidden"
      />

      {/* FLOATING TOOLBAR — fixed position, no layout impact */}
      <PACSToolbar
        onLogoClick={() => setShowLanding(true)}
        onUpload={() => fileInputRef.current?.click()}
        isMeasuring={isMeasuring}
        onMeasuringChange={setIsMeasuring}
        clippingY={clippingY}
        onClippingChange={setClippingY}
        onAnalyze={handleAnalyze}
        analysisStatus={analysisStatus}
        canAnalyze={originalObjUrl !== null}
        onExportReport={handleExportReport}
        canExport={analysisStatus === 'success'}
      />

      {/* LANDING PAGE MODAL */}
      {showLanding && (
        <LandingPage
          onOpenApp={() => setShowLanding(false)}
          onStartMesh={() => {
            setShowLanding(false);
            setUploadMode('mesh');
            if (originalObjUrl) handleReset();
            setTimeout(() => fileInputRef.current?.click(), 100);
          }}
          onStartDicom={() => {
            setShowLanding(false);
            setUploadMode('dicom');
            if (originalObjUrl) handleReset();
          }}
        />
      )}

      {/* MAIN CONTENT AREA — full width, toolbar floats over it */}
      <div className="flex-1 relative flex flex-col" style={{ minWidth: 0 }}>

        {/* Mode toggle — visible when no mesh is loaded yet */}
        {!originalObjUrl && (
          <div
            className="absolute top-5 left-1/2 -translate-x-1/2 flex"
            style={{
              background: 'rgba(0, 0, 0, 0.05)',
              backdropFilter: 'blur(24px)',
              borderRadius: 24,
              padding: 3,
              border: '1px solid rgba(0,0,0,0.08)',
              zIndex: 20,
            }}
          >
            {[
              { id: 'mesh', label: 'Upload mesh (.obj/.ply/.stl)' },
              { id: 'dicom', label: 'Upload DICOM/NIfTI scan' },
            ].map((m) => (
              <button
                key={m.id}
                onClick={() => setUploadMode(m.id)}
                style={{
                  padding: '6px 14px',
                  borderRadius: 20,
                  fontSize: 12,
                  fontWeight: 400,
                  border: 'none',
                  cursor: 'pointer',
                  background: uploadMode === m.id ? 'rgba(0,0,0,0.08)' : 'transparent',
                  color: uploadMode === m.id ? '#0F1117' : '#64748B',
                  transition: 'all 0.2s ease',
                }}
              >
                {m.label}
              </button>
            ))}
          </div>
        )}

        {/* Upload State */}
        {!originalObjUrl && uploadMode === 'dicom' ? (
          <DicomWorkflow
            apiUrl={API_URL}
            initialSessionId={dicomSessionId}
            onAnalysisComplete={(payload, { croppedMeshUrl, sessionId }) => {
              // Mirror the mesh-flow state updates so all downstream
              // viewers/HUD/report components render identically.
              setRiskScore(payload.risk_score);
              setRiskLevel(payload.risk_level);
              setInterpretation(payload.interpretation);
              setMorphologyData(payload.morphology ?? null);
              setClinicalReport(payload.clinical_report ?? null);
              setDicomAnalysis({
                harmonization: payload.harmonization ?? null,
                crop_info: payload.crop_info ?? null,
              });
              setDicomSessionId(sessionId);
              setAnalysisStatus('success');
              setWireframeMode(false);
              setViewMode('risk');
              setHeatmapData(null);

              // Load the cropped mesh into ArteryViewer — this unmounts
              // DicomWorkflow and activates the same post-analyze UI as
              // the .obj upload flow.
              setOriginalObjUrl(croppedMeshUrl);

              // Non-blocking heatmap fetch using the session-based GET.
              setHeatmapLoading(true);
              fetch(`${API_URL}/dicom/cropped-heatmap/${sessionId}`)
                .then((res) => (res.ok ? res.json() : null))
                .then((hData) => {
                  if (hData?.heatmap) setHeatmapData(hData.heatmap);
                  setHeatmapLoading(false);
                })
                .catch((err) => {
                  console.warn('Cropped heatmap fetch failed (non-critical):', err);
                  setHeatmapLoading(false);
                });
            }}
          />
        ) : !originalObjUrl ? (
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
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
                  Upload Vessel Mesh
                </div>
                <div style={{ fontSize: 14, color: '#64748B', lineHeight: 1.6, maxWidth: 360, margin: '0 auto' }}>
                  Load a 3D surface mesh of the aneurysm ROI for geometric analysis and rupture risk prediction.
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
                onClick={() => fileInputRef.current?.click()}
                style={{
                  border: '1.5px dashed rgba(220,38,38,0.2)',
                  borderRadius: 24,
                  padding: '48px 40px 44px',
                  textAlign: 'center',
                  cursor: 'pointer',
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
                  <Upload style={{ width: 28, height: 28, color: '#dc2626' }} />
                </motion.div>

                <div style={{ fontSize: 20, fontWeight: 700, color: '#000229', marginBottom: 8, letterSpacing: '-0.02em', fontFamily: 'Syne, sans-serif' }}>
                  {dragActive ? 'Release to upload' : 'Drop your mesh file here'}
                </div>
                <div style={{ fontSize: 13, color: '#94A3B8', lineHeight: 1.6 }}>
                  or click anywhere to browse your files
                </div>
              </motion.div>

              {/* Format badges + button row */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {['.obj', '.ply', '.stl'].map((ext) => (
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
                  onClick={() => fileInputRef.current?.click()}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 8,
                    padding: '10px 24px',
                    background: '#dc2626',
                    color: '#ffffff',
                    fontWeight: 600, fontSize: 13,
                    borderRadius: 10, border: 'none',
                    cursor: 'pointer',
                    boxShadow: '0 4px 14px rgba(220,38,38,0.3)',
                    whiteSpace: 'nowrap',
                    flexShrink: 0,
                    fontFamily: 'Syne, sans-serif',
                  }}
                >
                  <Upload style={{ width: 15, height: 15 }} />
                  Select File
                </button>
              </div>
            </motion.div>
          </div>
        ) : (
          <>
            {/* 3D VIEWPORT (fills remaining space) */}
            <div className="flex-1 relative" style={{ minHeight: 0 }}>
              <ArteryViewer
                ref={viewerRef}
                originalObjUrl={originalObjUrl}
                wireframeMode={wireframeMode}
                onWireframeModeChange={setWireframeMode}
                clippingY={clippingY}
                isMeasuring={isMeasuring}
                onMeasurementUpdate={setCurrentMeasurement}
                activeHighlight={activeHighlight}
                heatmapData={heatmapData}
                viewMode={viewMode}
                resetViewKey={resetViewKey}
              />

              {/* TOP BAR — transparent, floating elements */}
              <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-5"
                style={{ height: 56, zIndex: 40 }}>

                {/* Left: Back to DICOM only — file/reset now live in the left floating column */}
                <div className="flex items-center gap-2">
                  {dicomSessionId && (
                    <button
                      onClick={handleBackToDicom}
                      style={{
                        background: 'rgba(255,255,255,0.92)',
                        backdropFilter: 'blur(24px)',
                        borderRadius: 20,
                        padding: '6px 12px',
                        border: '1px solid rgba(0,0,0,0.08)',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                        color: '#94A3B8',
                        fontSize: 11,
                        fontWeight: 400,
                      }}
                      title="Return to the vessel tree and click a different region"
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M19 12H5" />
                        <path d="M12 19l-7-7 7-7" />
                      </svg>
                      Back to DICOM
                    </button>
                  )}
                </div>

                {/* Center: View mode segmented control */}
                {analysisStatus === 'success' && (
                  <ViewModeControl
                    viewMode={viewMode}
                    onViewModeChange={setViewMode}
                    heatmapData={heatmapData}
                    heatmapLoading={heatmapLoading}
                  />
                )}

                {/* Right spacer */}
                <div style={{ width: 80 }} />
              </div>

              {/* Status Indicators (measuring / clipping) */}
              <div className="absolute top-16 left-1/2 -translate-x-1/2 flex gap-2" style={{ zIndex: 30 }}>
                <MeasurementIndicator
                  isMeasuring={isMeasuring}
                  measurement={currentMeasurement}
                />
                <ClippingIndicator clippingY={clippingY} />
              </div>

              {/* Left column: Risk card (risk view only) + controls row — shown on all analysis views */}
              {analysisStatus === 'success' && (
                <div style={{ position: 'absolute', top: 8, left: 80, zIndex: 30, display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {/* Risk card — only on risk view */}
                  {viewMode === 'risk' && riskScore !== null && (
                    <RiskScoreCard riskScore={riskScore} riskLevel={riskLevel} />
                  )}

                  {/* Controls row — file name · wireframe · reset view */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    {/* File name + close */}
                    <div style={{
                      display: 'flex', alignItems: 'center', gap: 8,
                      background: 'rgba(255,255,255,0.92)', backdropFilter: 'blur(24px)',
                      WebkitBackdropFilter: 'blur(24px)', borderRadius: 20,
                      padding: '6px 12px', border: '1px solid rgba(0,0,0,0.08)',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
                    }}>
                      <span style={{ fontSize: 11, color: '#94A3B8', fontWeight: 400, maxWidth: 110, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {file?.name?.replace('.obj', '') || 'Unknown'}
                      </span>
                      <button onClick={handleReset} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0, display: 'flex', color: '#475569' }} title="Close scan">
                        <X style={{ width: 13, height: 13 }} />
                      </button>
                    </div>

                    {/* Wireframe toggle */}
                    <WireframeToggle wireframe={wireframeMode} onChange={setWireframeMode} />

                    {/* Reset view */}
                    <button
                      onClick={() => setResetViewKey(k => k + 1)}
                      style={{
                        background: 'rgba(255,255,255,0.92)', backdropFilter: 'blur(24px)',
                        WebkitBackdropFilter: 'blur(24px)', borderRadius: 20,
                        padding: '6px 10px', border: '1px solid rgba(0,0,0,0.08)',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
                        cursor: 'pointer', display: 'flex', alignItems: 'center', color: '#64748B',
                      }}
                      title="Reset view"
                    >
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M15 3h6v6" /><path d="M9 21H3v-6" />
                        <path d="M21 3l-7 7" /><path d="M3 21l7-7" />
                      </svg>
                    </button>
                  </div>
                </div>
              )}

              {/* Loading Overlay */}
              <AnimatePresence>
                {analysisStatus === 'loading' && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 flex items-center justify-center pointer-events-none"
                    style={{ background: 'rgba(255,255,255,0.88)', backdropFilter: 'blur(4px)', zIndex: 50 }}
                  >
                    <div className="text-center">
                      <Loader2 style={{ width: 40, height: 40, color: '#dc2626', animation: 'spin 1s linear infinite', margin: '0 auto 16px' }} />
                      <p style={{ color: '#64748B', fontSize: 12, fontWeight: 400, textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                        Analyzing morphology...
                      </p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Export Toast */}
              <AnimatePresence>
                {showExportToast && (
                  <motion.div
                    initial={{ opacity: 0, y: 50 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 50 }}
                    className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-2.5"
                    style={{
                      background: 'rgba(5, 150, 105, 0.9)',
                      backdropFilter: 'blur(12px)',
                      borderRadius: 10,
                      padding: '8px 16px',
                      zIndex: 50,
                    }}
                  >
                    <span style={{ color: '#fff', fontSize: 13 }}>Report exported successfully</span>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Error Toast */}
              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: 50 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 50 }}
                    className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-2.5"
                    style={{
                      background: 'rgba(239, 68, 68, 0.9)',
                      backdropFilter: 'blur(12px)',
                      borderRadius: 10,
                      padding: '8px 16px',
                      zIndex: 50,
                    }}
                  >
                    <AlertTriangle style={{ width: 14, height: 14, color: '#fff' }} />
                    <span style={{ color: '#fff', fontSize: 13 }}>{error}</span>
                    <button
                      onClick={() => setError(null)}
                      style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.7)', cursor: 'pointer', marginLeft: 4 }}
                    >
                      <X style={{ width: 14, height: 14 }} />
                    </button>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* FLOATING MORPHOLOGY PANEL — right side, overlays the 3D scene */}
              <AnimatePresence>
                {((originalObjUrl || dicomAnalysis) && (analysisStatus === 'loading' || clinicalReport)) && (
                  <div style={{ position: 'absolute', top: 8, right: 20, zIndex: 30, display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {dicomAnalysis && (
                      <div
                        style={{
                          display: 'flex', alignItems: 'center', gap: 6, alignSelf: 'flex-end',
                          background: dicomAnalysis.harmonization?.all_in_distribution
                            ? 'rgba(255,255,255,0.88)' : 'rgba(255,255,255,0.88)',
                          backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)',
                          border: `1px solid ${dicomAnalysis.harmonization?.all_in_distribution ? 'rgba(52,211,153,0.4)' : 'rgba(234,179,8,0.4)'}`,
                          borderRadius: 20, padding: '5px 12px',
                          boxShadow: '0 4px 16px rgba(0,0,0,0.08)',
                          fontSize: 10, fontWeight: 500,
                          color: dicomAnalysis.harmonization?.all_in_distribution ? '#059669' : '#b45309',
                        }}
                        title="Input quality vs. training distribution"
                      >
                        {dicomAnalysis.harmonization?.all_in_distribution ? (
                          <><CheckCircle style={{ width: 11, height: 11 }} /> In distribution</>
                        ) : (
                          <><AlertTriangle style={{ width: 11, height: 11 }} /> Out of distribution</>
                        )}
                      </div>
                    )}
                    <MorphologyReport
                      clinicalReport={clinicalReport}
                      onParameterSelect={handleParameterSelect}
                      isLoading={analysisStatus === 'loading'}
                    />
                  </div>
                )}
              </AnimatePresence>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default App;
