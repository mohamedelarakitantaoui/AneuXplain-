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
      background: 'rgba(26, 29, 39, 0.9)',
      backdropFilter: 'blur(24px)',
      borderRadius: 24,
      padding: 3,
      border: '1px solid rgba(255,255,255,0.06)',
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
              ? 'rgba(255,255,255,0.1)'
              : 'transparent',
            color: viewMode === mode.id
              ? '#F1F5F9'
              : mode.disabled
                ? '#374151'
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
        border: '1px solid rgba(255,255,255,0.06)',
        background: wireframe ? 'rgba(74, 158, 255, 0.1)' : 'rgba(26, 29, 39, 0.9)',
        backdropFilter: 'blur(24px)',
        cursor: 'pointer',
        fontSize: 11,
        color: wireframe ? '#4A9EFF' : '#64748B',
        transition: 'all 0.2s ease',
      }}
    >
      <div style={{
        width: 24,
        height: 14,
        borderRadius: 7,
        background: wireframe ? '#4A9EFF' : '#374151',
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
      style={{ background: '#0F1117', color: '#F1F5F9', fontFamily: "'Inter', system-ui, sans-serif" }}
    >
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".obj"
        onChange={(e) => e.target.files[0] && handleUpload(e.target.files[0])}
        className="hidden"
      />

      {/* LEFT TOOLBAR (56px) */}
      <PACSToolbar
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

      {/* MAIN CONTENT AREA */}
      <div className="flex-1 relative flex flex-col" style={{ minWidth: 0 }}>

        {/* Mode toggle — visible when no mesh is loaded yet */}
        {!originalObjUrl && (
          <div
            className="absolute top-5 left-1/2 -translate-x-1/2 flex"
            style={{
              background: 'rgba(26, 29, 39, 0.9)',
              backdropFilter: 'blur(24px)',
              borderRadius: 24,
              padding: 3,
              border: '1px solid rgba(255,255,255,0.06)',
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
                  background: uploadMode === m.id ? 'rgba(255,255,255,0.1)' : 'transparent',
                  color: uploadMode === m.id ? '#F1F5F9' : '#64748B',
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
            className={`absolute inset-0 flex flex-col items-center justify-center transition-all duration-300 ${
              dragActive ? 'ring-2 ring-inset ring-[#4A9EFF]' : ''
            }`}
            style={{ background: 'linear-gradient(135deg, #0F1117 0%, #141821 50%, #0F1117 100%)' }}
          >
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center"
            >
              {/* Branding */}
              <div className="flex items-center justify-center gap-4 mb-12">
                <div className="w-12 h-12 rounded-xl flex items-center justify-center"
                  style={{ background: 'linear-gradient(135deg, #4A9EFF, #3B82F6)', boxShadow: '0 8px 24px rgba(74, 158, 255, 0.2)' }}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 2L2 7l10 5 10-5-10-5z" />
                    <path d="M2 17l10 5 10-5" />
                    <path d="M2 12l10 5 10-5" />
                  </svg>
                </div>
                <div className="text-left">
                  <h1 style={{ fontSize: 28, fontWeight: 300, letterSpacing: '-0.02em', color: '#F1F5F9' }}>AneuXplain</h1>
                  <p style={{ fontSize: 12, color: '#64748B', fontWeight: 400, letterSpacing: '0.02em' }}>Explainable AI for Intracranial Aneurysm Rupture Risk Prediction</p>
                </div>
              </div>

              <div className="w-28 h-28 mx-auto mb-8 flex items-center justify-center"
                style={{ background: 'rgba(26, 29, 39, 0.6)', borderRadius: 20, border: '1px dashed rgba(255,255,255,0.08)' }}>
                <Upload style={{ width: 48, height: 48, color: '#374151' }} />
              </div>
              <h3 style={{ fontSize: 22, fontWeight: 300, color: '#F1F5F9', marginBottom: 8, letterSpacing: '-0.01em' }}>
                Drop ROI Mesh Here
              </h3>
              <p style={{ color: '#64748B', marginBottom: 36, maxWidth: 400, fontSize: 13, fontWeight: 400, lineHeight: 1.6 }}>
                Cropped aneurysm region (.obj format) — drag and drop or click to browse
              </p>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center gap-2.5 mx-auto"
                style={{
                  padding: '12px 28px',
                  background: 'linear-gradient(135deg, #4A9EFF, #3B82F6)',
                  color: '#fff',
                  fontWeight: 500,
                  fontSize: 14,
                  borderRadius: 10,
                  border: 'none',
                  cursor: 'pointer',
                  boxShadow: '0 4px 16px rgba(74, 158, 255, 0.25)',
                  transition: 'all 0.2s ease',
                }}
              >
                <Upload style={{ width: 18, height: 18 }} />
                Select File
              </button>
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

                {/* Left: Patient name pill + reset view + close */}
                <div className="flex items-center gap-2">
                  <div
                    className="flex items-center gap-2.5"
                    style={{
                      background: 'rgba(26, 29, 39, 0.9)',
                      backdropFilter: 'blur(24px)',
                      borderRadius: 20,
                      padding: '6px 14px',
                      border: '1px solid rgba(255,255,255,0.06)',
                    }}
                  >
                    <span style={{ fontSize: 12, color: '#94A3B8', fontWeight: 400 }}>
                      {file?.name?.replace('.obj', '') || 'Unknown'}
                    </span>
                    <button
                      onClick={handleReset}
                      style={{
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        padding: 0,
                        display: 'flex',
                        color: '#475569',
                      }}
                      title="Close scan"
                    >
                      <X style={{ width: 14, height: 14 }} />
                    </button>
                  </div>
                  {/* Back to DICOM — only when analysis came from the DICOM flow */}
                  {dicomSessionId && (
                    <button
                      onClick={handleBackToDicom}
                      style={{
                        background: 'rgba(26, 29, 39, 0.9)',
                        backdropFilter: 'blur(24px)',
                        borderRadius: 20,
                        padding: '6px 12px',
                        border: '1px solid rgba(255,255,255,0.06)',
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
                  {/* Reset View button */}
                  <button
                    onClick={() => setResetViewKey(k => k + 1)}
                    style={{
                      background: 'rgba(26, 29, 39, 0.9)',
                      backdropFilter: 'blur(24px)',
                      borderRadius: 20,
                      padding: '6px 10px',
                      border: '1px solid rgba(255,255,255,0.06)',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      color: '#64748B',
                      transition: 'color 0.2s ease',
                    }}
                    title="Reset view"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M15 3h6v6" />
                      <path d="M9 21H3v-6" />
                      <path d="M21 3l-7 7" />
                      <path d="M3 21l7-7" />
                    </svg>
                  </button>
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

                {/* Right: Wireframe toggle */}
                {analysisStatus === 'success' && (viewMode === 'risk' || viewMode === 'measurements') && (
                  <WireframeToggle
                    wireframe={wireframeMode}
                    onChange={setWireframeMode}
                  />
                )}

                {/* Spacer when wireframe toggle is hidden */}
                {!(analysisStatus === 'success' && (viewMode === 'risk' || viewMode === 'measurements')) && (
                  <div style={{ width: 80 }} />
                )}
              </div>

              {/* Status Indicators (measuring / clipping) */}
              <div className="absolute top-16 left-1/2 -translate-x-1/2 flex gap-2" style={{ zIndex: 30 }}>
                <MeasurementIndicator
                  isMeasuring={isMeasuring}
                  measurement={currentMeasurement}
                />
                <ClippingIndicator clippingY={clippingY} />
              </div>

              {/* Risk Score Card — top-right floating */}
              {viewMode === 'risk' && riskScore !== null && (
                <div className="absolute top-16 right-5" style={{ zIndex: 30 }}>
                  <RiskScoreCard
                    riskScore={riskScore}
                    riskLevel={riskLevel}
                  />
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
                    style={{ background: 'rgba(15, 17, 23, 0.88)', backdropFilter: 'blur(4px)', zIndex: 50 }}
                  >
                    <div className="text-center">
                      <Loader2 style={{ width: 40, height: 40, color: '#4A9EFF', animation: 'spin 1s linear infinite', margin: '0 auto 16px' }} />
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
            </div>
          </>
        )}
      </div>

      {/* RIGHT PANEL — Morphology Report (slides in after analysis) */}
      <AnimatePresence>
        {(originalObjUrl && (analysisStatus === 'loading' || clinicalReport)) ||
        (dicomAnalysis && clinicalReport) ? (
          <div className="relative">
            {dicomAnalysis && (
              <div
                className="absolute top-3 right-3 flex items-center gap-1.5"
                style={{
                  background: dicomAnalysis.harmonization?.all_in_distribution
                    ? 'rgba(5, 150, 105, 0.15)'
                    : 'rgba(234, 179, 8, 0.15)',
                  border: `1px solid ${
                    dicomAnalysis.harmonization?.all_in_distribution
                      ? 'rgba(5, 150, 105, 0.4)'
                      : 'rgba(234, 179, 8, 0.4)'
                  }`,
                  borderRadius: 14,
                  padding: '4px 10px',
                  zIndex: 50,
                  fontSize: 10,
                  fontWeight: 500,
                  color: dicomAnalysis.harmonization?.all_in_distribution ? '#34d399' : '#facc15',
                }}
                title="Input quality vs. training distribution"
              >
                {dicomAnalysis.harmonization?.all_in_distribution ? (
                  <>
                    <CheckCircle style={{ width: 11, height: 11 }} />
                    In distribution
                  </>
                ) : (
                  <>
                    <AlertTriangle style={{ width: 11, height: 11 }} />
                    Out of distribution
                  </>
                )}
              </div>
            )}
            <MorphologyReport
              clinicalReport={clinicalReport}
              onParameterSelect={handleParameterSelect}
              isLoading={analysisStatus === 'loading'}
            />
          </div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}

export default App;
