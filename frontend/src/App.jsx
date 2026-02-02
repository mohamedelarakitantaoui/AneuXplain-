import { useState, useCallback, useRef, useMemo, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  Activity,
  AlertTriangle,
  Loader2,
  X,
  Layers
} from 'lucide-react';
import ArteryViewer from './components/ArteryViewer';
import { PACSToolbar, MorphControlBar, MeasurementIndicator, ClippingIndicator } from './components/Overlay';
import RiskAnalysisHUD from './components/RiskAnalysisHUD';

const API_URL = 'http://localhost:8000';


// ============================================
// MAIN APP COMPONENT
// ============================================
function App() {
  // State
  const [activeTab, setActiveTab] = useState('home');
  const [file, setFile] = useState(null);
  const [originalObjUrl, setOriginalObjUrl] = useState(null);
  const [healedObjUrl, setHealedObjUrl] = useState(null);
  const [riskScore, setRiskScore] = useState(null);
  const [riskLevel, setRiskLevel] = useState(null);
  const [interpretation, setInterpretation] = useState(null);
  const [analysisStatus, setAnalysisStatus] = useState('idle'); // idle, loading, success, error
  const [healingStatus, setHealingStatus] = useState('idle');
  const [healData, setHealData] = useState(null);
  const [error, setError] = useState(null);
  const [showComparison, setShowComparison] = useState(false);
  const [wireframeMode, setWireframeMode] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  // Medical Workstation Tools State
  const [clippingY, setClippingY] = useState(2); // 2 = no clipping (above model)
  const [isMeasuring, setIsMeasuring] = useState(false);
  const [currentMeasurement, setCurrentMeasurement] = useState(null);

  // Manifold Morphing State (THESIS FEATURES)
  const [morphValue, setMorphValue] = useState(0); // 0 = original, 1 = healed
  const [showHeatmap, setShowHeatmap] = useState(false); // Root cause visualization

  const fileInputRef = useRef(null);

  // ============================================
  // COMPUTED VALUES: Interpolated Risk for Real-time Update
  // ============================================
  const interpolatedRisk = useMemo(() => {
    if (riskScore === null) return null;
    if (!healData || healData.final_risk === undefined) return riskScore;
    // Linear interpolation: riskScore → healData.final_risk as morphValue goes 0→1
    return riskScore + (healData.final_risk - riskScore) * morphValue;
  }, [riskScore, healData, morphValue]);

  // Show "(Est.)" when user is actively morphing (not at 0 or 1)
  const isEstimate = morphValue > 0 && morphValue < 1 && healData !== null;

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

  // Upload file only (no auto-analysis)
  const handleUpload = async (selectedFile) => {
    setFile(selectedFile);
    setError(null);
    setRiskScore(null);
    setRiskLevel(null);
    setInterpretation(null);
    setHealData(null);
    setHealedObjUrl(null);
    setShowComparison(false);
    setWireframeMode(false);
    setAnalysisStatus('idle');
    setHealingStatus('idle');

    // Create object URL for 3D viewer
    const objUrl = URL.createObjectURL(selectedFile);
    setOriginalObjUrl(objUrl);
  };

  // Analyze geometry (Step 1)
  const handleAnalyze = async () => {
    if (!file) return;

    setError(null);
    setAnalysisStatus('loading');

    const formData = new FormData();
    formData.append('file', file);

    try {
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
      setAnalysisStatus('success');
    } catch (err) {
      setError(err.message);
      setAnalysisStatus('error');
      console.error('Analysis error:', err);
    }
  };

  // Heal artery (Generate Counterfactual - Step 2)
  const handleHeal = async () => {
    if (!file) return;

    setHealingStatus('loading');
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Get healed OBJ file
      const response = await fetch(`${API_URL}/heal?return_file=true`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Healing failed');
      }

      const blob = await response.blob();
      const healedUrl = URL.createObjectURL(blob);
      setHealedObjUrl(healedUrl);
      setShowComparison(true);

      // Get stats
      const statsFormData = new FormData();
      statsFormData.append('file', file);

      const statsResponse = await fetch(`${API_URL}/heal`, {
        method: 'POST',
        body: statsFormData,
      });

      if (statsResponse.ok) {
        const stats = await statsResponse.json();
        // Use real geometric deltas from the API (Phase 5: Interpretation)
        setHealData(stats);
      }

      setHealingStatus('success');
    } catch (err) {
      setError(err.message);
      setHealingStatus('error');
      console.error('Healing error:', err);
    }
  };

  // Export Report (Step 3)
  const handleExportReport = () => {
    const report = {
      timestamp: new Date().toISOString(),
      filename: file?.name || 'unknown',
      analysis: {
        risk_score: riskScore,
        risk_level: riskLevel,
        interpretation: interpretation,
      },
      counterfactual: healData ? {
        initial_risk: healData.initial_risk,
        final_risk: healData.final_risk,
        risk_reduction_pct: healData.risk_reduction_pct,
        success: healData.success,
        steps_taken: healData.steps_taken,
        geometric_deltas: {
          max_displacement_mm: healData.max_displacement_mm,
          mean_displacement_mm: healData.mean_displacement_mm,
          displacement_std_mm: healData.displacement_std_mm,
          volume_change_pct: healData.volume_change_pct,
          surface_area_change_pct: healData.surface_area_change_pct,
        },
      } : null,
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `aneurysm_report_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Reset
  const handleReset = () => {
    setFile(null);
    setOriginalObjUrl(null);
    setHealedObjUrl(null);
    setRiskScore(null);
    setRiskLevel(null);
    setInterpretation(null);
    setHealData(null);
    setAnalysisStatus('idle');
    setHealingStatus('idle');
    setError(null);
    setShowComparison(false);
    setWireframeMode(false);
  };

  // ============================================
  // RENDER - PACS WORKSTATION LAYOUT
  // ============================================

  return (
    <div className="h-screen w-screen bg-slate-950 text-white overflow-hidden flex font-sans">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".obj"
        onChange={(e) => e.target.files[0] && handleUpload(e.target.files[0])}
        className="hidden"
      />

      {/* ============================================
          LEFT SIDEBAR - PACS Toolbar (w-16)
          ============================================ */}
      <PACSToolbar
        onUpload={() => fileInputRef.current?.click()}
        isMeasuring={isMeasuring}
        onMeasuringChange={setIsMeasuring}
        clippingY={clippingY}
        onClippingChange={setClippingY}
        showHeatmap={showHeatmap}
        onHeatmapChange={setShowHeatmap}
        showHeatmapToggle={showComparison && healedObjUrl !== null}
        onAnalyze={handleAnalyze}
        analysisStatus={analysisStatus}
        canAnalyze={originalObjUrl !== null}
      />

      {/* ============================================
          MAIN CANVAS AREA (flex-1)
          ============================================ */}
      <div className="flex-1 relative">
        {/* Upload State - Centered overlay */}
        {!originalObjUrl ? (
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            className={`absolute inset-0 flex flex-col items-center justify-center transition-all duration-300 bg-linear-to-br from-slate-950 via-slate-900 to-slate-950 ${dragActive
              ? 'ring-2 ring-inset ring-cyan-500 bg-cyan-500/5'
              : ''
              }`}
          >
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center"
            >
              {/* Branding */}
              <div className="flex items-center justify-center gap-4 mb-12">
                <div className="w-14 h-14 bg-linear-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-cyan-500/20">
                  <Layers className="w-7 h-7 text-white" />
                </div>
                <div className="text-left">
                  <h1 className="text-3xl font-light tracking-tight text-white">NeuroTwin</h1>
                  <p className="text-sm text-slate-500 font-light tracking-wide">Counterfactual Optimization Engine</p>
                </div>
              </div>

              <div className="w-32 h-32 mx-auto mb-8 bg-slate-900/50 rounded-2xl flex items-center justify-center border border-slate-800 border-dashed">
                <Upload className="w-14 h-14 text-slate-600" />
              </div>
              <h3 className="text-2xl font-light text-white mb-3 tracking-tight">
                Drop ROI Mesh Here
              </h3>
              <p className="text-slate-500 mb-10 max-w-md font-light">
                Cropped aneurysm region (.obj format) — drag and drop or click to browse
              </p>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="px-8 py-4 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-medium rounded-lg transition-all duration-300 flex items-center gap-3 mx-auto shadow-lg shadow-cyan-500/20"
              >
                <Upload className="w-5 h-5" />
                Select File
              </button>
            </motion.div>
          </div>
        ) : (
          <>
            {/* FULL-SCREEN 3D CANVAS */}
            <ArteryViewer
              originalObjUrl={originalObjUrl}
              healedObjUrl={healedObjUrl}
              showComparison={showComparison}
              wireframeMode={wireframeMode}
              onWireframeModeChange={setWireframeMode}
              clippingY={clippingY}
              isMeasuring={isMeasuring}
              onMeasurementUpdate={setCurrentMeasurement}
              morphValue={morphValue}
              onMorphValueChange={setMorphValue}
              showHeatmap={showHeatmap}
              onHeatmapChange={setShowHeatmap}
            />

            {/* ============================================
                OVERLAY UI ELEMENTS
                ============================================ */}

            {/* TOP-LEFT: Patient/File Info + Reset */}
            <div className="absolute top-8 left-8 flex items-center gap-3">
              <div className="bg-slate-900/95 backdrop-blur-xl rounded-2xl border border-slate-700/40 px-5 py-3.5 flex items-center gap-4 shadow-xl shadow-black/20">
                <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-cyan-500/20">
                  <Layers className="w-5 h-5 text-white" />
                </div>
                <div>
                  <p className="text-[10px] text-slate-500 uppercase tracking-[0.2em] font-medium">PATIENT SCAN</p>
                  <p className="text-sm text-white font-light mt-0.5">{file?.name?.replace('.obj', '') || 'Unknown'}</p>
                </div>
              </div>
              <button
                onClick={handleReset}
                className="bg-slate-900/95 backdrop-blur-xl rounded-2xl border border-slate-700/40 p-3 text-slate-500 hover:text-white hover:border-slate-500 transition-all shadow-lg shadow-black/20"
                title="New Scan"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* TOP-RIGHT: Futuristic Risk Analysis HUD */}
            <div className="absolute top-8 right-12 z-50">
              <RiskAnalysisHUD
                riskScore={riskScore}
                riskLevel={riskLevel}
                healData={healData}
                visible={riskScore !== null || healData !== null}
              />
            </div>

            {/* TOP-CENTER: Status Indicators */}
            <div className="absolute top-8 left-1/2 -translate-x-1/2 flex gap-3">
              <MeasurementIndicator
                isMeasuring={isMeasuring}
                measurement={currentMeasurement}
              />
              <ClippingIndicator clippingY={clippingY} />
            </div>

            {/* ============================================
                BOTTOM CONTROL BAR - Pre-Op/Post-Op Slider
                ============================================ */}
            <div className="absolute bottom-10 left-1/2 -translate-x-1/2">
              <MorphControlBar
                morphValue={morphValue}
                onMorphChange={setMorphValue}
                visible={showComparison && healedObjUrl !== null}
                onHeal={handleHeal}
                healingStatus={healingStatus}
                canHeal={analysisStatus === 'success' && healingStatus !== 'success'}
              />
            </div>

            {/* Loading Overlay */}
            <AnimatePresence>
              {(analysisStatus === 'loading' || healingStatus === 'loading') && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="absolute inset-0 bg-slate-950/90 backdrop-blur-sm flex items-center justify-center pointer-events-none"
                >
                  <div className="text-center">
                    <Loader2 className="w-12 h-12 text-cyan-400 animate-spin mx-auto mb-4" />
                    <p className="text-slate-400 text-sm font-light tracking-wide uppercase">
                      {analysisStatus === 'loading'
                        ? 'Sampling Surface Manifold...'
                        : 'Computing Risk Gradient...'}
                    </p>
                  </div>
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
                  className="absolute bottom-24 left-1/2 -translate-x-1/2 bg-red-500/90 backdrop-blur-sm rounded-lg px-5 py-3 flex items-center gap-3"
                >
                  <AlertTriangle className="w-4 h-4 text-white" />
                  <span className="text-white text-sm">{error}</span>
                  <button
                    onClick={() => setError(null)}
                    className="text-white/70 hover:text-white ml-2"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </>
        )}
      </div>
    </div>
  );
}

export default App;
