import { Ruler, Scissors, RotateCcw, Layers, Sparkles, Target } from 'lucide-react';

/**
 * MedicalToolsSidebar - Professional tools panel for the Medical Workstation
 * 
 * Features:
 * - Clipping Plane (Y-axis slice) slider
 * - Measurement mode toggle
 * - Current measurement display
 * - Manifold Explorer (Morph Slider) - NEW
 * - Root Cause Heatmap Toggle - NEW
 * 
 * Usage:
 * <MedicalToolsSidebar
 *   clippingY={clippingY}
 *   onClippingChange={setClippingY}
 *   isMeasuring={isMeasuring}
 *   onMeasuringChange={setIsMeasuring}
 *   currentMeasurement={measurement}
 *   morphValue={morphValue}
 *   onMorphChange={setMorphValue}
 *   showHeatmap={showHeatmap}
 *   onHeatmapChange={setShowHeatmap}
 *   showManifoldTools={showComparison}
 * />
 */
export function MedicalToolsSidebar({
  clippingY = 2,
  onClippingChange,
  isMeasuring = false,
  onMeasuringChange,
  currentMeasurement = null,
  onResetView,
  // NEW: Manifold Learning Props
  morphValue = 0,
  onMorphChange,
  showHeatmap = false,
  onHeatmapChange,
  showManifoldTools = false,
}) {
  // Convert clippingY to percentage for display (2 = 100% visible, -2 = 0%)
  const clippingPercent = Math.round(((clippingY + 2) / 4) * 100);
  
  // Get risk state label based on morph value
  const getRiskState = (t) => {
    if (t < 0.3) return { label: 'HIGH RISK', color: 'text-red-400', bg: 'bg-red-500/20' };
    if (t < 0.7) return { label: 'TRANSITIONING', color: 'text-yellow-400', bg: 'bg-yellow-500/20' };
    return { label: 'LOW RISK', color: 'text-green-400', bg: 'bg-green-500/20' };
  };
  
  const riskState = getRiskState(morphValue);

  return (
    <div className="bg-slate-800/80 backdrop-blur-sm rounded-xl border border-slate-700/50 p-4 space-y-6">
      {/* Header */}
      <div className="flex items-center gap-2 pb-3 border-b border-slate-700/50">
        <Layers className="w-5 h-5 text-cyan-400" />
        <h3 className="text-sm font-semibold text-slate-200">Medical Tools</h3>
      </div>

      {/* Clipping Plane Control */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Scissors className="w-4 h-4 text-cyan-400" />
            <span className="text-sm text-slate-300">Clipping Plane</span>
          </div>
          <span className="text-xs text-slate-500 font-mono">
            {clippingY >= 2 ? 'OFF' : `${clippingPercent}%`}
          </span>
        </div>
        
        <div className="relative">
          <input
            type="range"
            min="-2"
            max="2"
            step="0.1"
            value={clippingY}
            onChange={(e) => onClippingChange?.(parseFloat(e.target.value))}
            className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer
                       [&::-webkit-slider-thumb]:appearance-none
                       [&::-webkit-slider-thumb]:w-4
                       [&::-webkit-slider-thumb]:h-4
                       [&::-webkit-slider-thumb]:bg-cyan-500
                       [&::-webkit-slider-thumb]:rounded-full
                       [&::-webkit-slider-thumb]:cursor-pointer
                       [&::-webkit-slider-thumb]:shadow-lg
                       [&::-webkit-slider-thumb]:shadow-cyan-500/30
                       [&::-webkit-slider-thumb]:transition-all
                       [&::-webkit-slider-thumb]:hover:bg-cyan-400
                       [&::-moz-range-thumb]:w-4
                       [&::-moz-range-thumb]:h-4
                       [&::-moz-range-thumb]:bg-cyan-500
                       [&::-moz-range-thumb]:rounded-full
                       [&::-moz-range-thumb]:border-0
                       [&::-moz-range-thumb]:cursor-pointer"
          />
          <div className="flex justify-between text-xs text-slate-600 mt-1">
            <span>Bottom</span>
            <span>Top</span>
          </div>
        </div>

        <p className="text-xs text-slate-500 italic">
          Slide to slice through the model and view internal structures
        </p>
      </div>

      {/* Divider */}
      <div className="border-t border-slate-700/50" />

      {/* Measurement Tool */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Ruler className="w-4 h-4 text-amber-400" />
            <span className="text-sm text-slate-300">Measurement</span>
          </div>
        </div>

        <button
          onClick={() => onMeasuringChange?.(!isMeasuring)}
          className={`w-full px-4 py-2.5 rounded-lg text-sm font-medium transition-all flex items-center justify-center gap-2 ${
            isMeasuring
              ? 'bg-amber-500 text-slate-900 shadow-lg shadow-amber-500/30'
              : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
          }`}
        >
          <Ruler className="w-4 h-4" />
          {isMeasuring ? 'Measuring Active' : 'Start Measuring'}
        </button>

        {/* Current Measurement Display */}
        {currentMeasurement !== null && (
          <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3">
            <div className="flex items-center justify-between">
              <span className="text-xs text-amber-400 font-medium">Distance</span>
              <span className="text-lg font-bold text-amber-300 font-mono">
                {currentMeasurement.toFixed(2)} mm
              </span>
            </div>
          </div>
        )}

        <p className="text-xs text-slate-500 italic">
          {isMeasuring 
            ? 'Click two points on the model to measure distance'
            : 'Enable to measure distances on the 3D model'}
        </p>
      </div>

      {/* Divider */}
      <div className="border-t border-slate-700/50" />

      {/* Quick Actions */}
      <div className="space-y-2">
        <button
          onClick={() => {
            onClippingChange?.(2);
            onMeasuringChange?.(false);
            onResetView?.();
          }}
          className="w-full px-4 py-2 rounded-lg text-sm font-medium bg-slate-700/50 text-slate-400 hover:bg-slate-600 hover:text-slate-200 transition-all flex items-center justify-center gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          Reset All Tools
        </button>
      </div>

      {/* ============================================
          MANIFOLD EXPLORER SECTION (THESIS FEATURE)
          Shows only when healed mesh is available
          ============================================ */}
      {showManifoldTools && (
        <>
          {/* Divider */}
          <div className="border-t border-cyan-500/30 mt-2" />

          {/* Manifold Explorer Header */}
          <div className="flex items-center gap-2 pt-2">
            <Sparkles className="w-5 h-5 text-cyan-400" />
            <h3 className="text-sm font-semibold text-cyan-300">Manifold Explorer</h3>
          </div>

          {/* Morph Slider - The Manifold */}
          <div className="space-y-3 bg-gradient-to-br from-cyan-500/10 to-purple-500/10 rounded-lg p-3 border border-cyan-500/20">
            <div className="flex justify-between text-xs">
              <span className="text-red-400 font-medium">🔴 Sick</span>
              <span className={`font-bold ${riskState.color} ${riskState.bg} px-2 py-0.5 rounded`}>
                {riskState.label}
              </span>
              <span className="text-green-400 font-medium">🟢 Healed</span>
            </div>
            
            {/* Morph Slider */}
            <div className="relative">
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={morphValue}
                onChange={(e) => onMorphChange?.(parseFloat(e.target.value))}
                className="w-full h-3 rounded-full appearance-none cursor-pointer
                           [&::-webkit-slider-thumb]:appearance-none
                           [&::-webkit-slider-thumb]:w-5
                           [&::-webkit-slider-thumb]:h-5
                           [&::-webkit-slider-thumb]:bg-white
                           [&::-webkit-slider-thumb]:rounded-full
                           [&::-webkit-slider-thumb]:shadow-lg
                           [&::-webkit-slider-thumb]:cursor-pointer
                           [&::-webkit-slider-thumb]:border-2
                           [&::-webkit-slider-thumb]:border-slate-300
                           [&::-moz-range-thumb]:w-5
                           [&::-moz-range-thumb]:h-5
                           [&::-moz-range-thumb]:bg-white
                           [&::-moz-range-thumb]:rounded-full
                           [&::-moz-range-thumb]:border-0
                           [&::-moz-range-thumb]:cursor-pointer"
                style={{
                  background: `linear-gradient(to right, 
                    #ef4444 0%, 
                    #f97316 25%,
                    #eab308 50%, 
                    #84cc16 75%,
                    #22c55e 100%)`
                }}
              />
            </div>

            {/* Value Display */}
            <div className="flex justify-between items-center text-xs">
              <div className="flex items-center gap-1">
                <span className="text-slate-400">t =</span>
                <span className={`font-mono font-bold ${riskState.color}`}>
                  {morphValue.toFixed(2)}
                </span>
              </div>
              <span className="text-slate-500 italic text-xs">Manifold Position</span>
            </div>

            <p className="text-xs text-cyan-300/60 italic text-center border-t border-slate-700 pt-2">
              ✨ Drag to traverse the geometric manifold
            </p>
          </div>

          {/* Root Cause Heatmap Toggle */}
          <div className="space-y-2 bg-purple-500/10 rounded-lg p-3 border border-purple-500/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4 text-purple-400" />
                <span className="text-sm text-slate-300 font-medium">Root Cause Heatmap</span>
              </div>
              <button
                onClick={() => onHeatmapChange?.(!showHeatmap)}
                className={`relative w-11 h-6 rounded-full transition-all duration-300 ${
                  showHeatmap 
                    ? 'bg-purple-500' 
                    : 'bg-slate-700'
                }`}
              >
                <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all duration-300 shadow ${
                  showHeatmap ? 'left-6' : 'left-1'
                }`} />
              </button>
            </div>
            
            {showHeatmap && (
              <div className="mt-2">
                <div className="w-full h-2 rounded" 
                  style={{ 
                    background: 'linear-gradient(to right, #3b82f6, #06b6d4, #10b981, #eab308, #ef4444)' 
                  }} 
                />
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                  <span>No Change</span>
                  <span>High Change</span>
                </div>
              </div>
            )}

            <p className="text-xs text-purple-300/60 italic">
              🎯 Isolate the geometric feature causing risk
            </p>
          </div>
        </>
      )}
    </div>
  );
}

/**
 * MedicalToolsCompact - A more compact version for smaller spaces
 */
export function MedicalToolsCompact({
  clippingY = 2,
  onClippingChange,
  isMeasuring = false,
  onMeasuringChange,
  currentMeasurement = null,
}) {
  return (
    <div className="flex items-center gap-4 bg-slate-800/80 backdrop-blur-sm rounded-lg px-4 py-2 border border-slate-700/50">
      {/* Clipping Control */}
      <div className="flex items-center gap-2">
        <Scissors className="w-4 h-4 text-cyan-400" />
        <input
          type="range"
          min="-2"
          max="2"
          step="0.1"
          value={clippingY}
          onChange={(e) => onClippingChange?.(parseFloat(e.target.value))}
          className="w-24 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer
                     [&::-webkit-slider-thumb]:appearance-none
                     [&::-webkit-slider-thumb]:w-3
                     [&::-webkit-slider-thumb]:h-3
                     [&::-webkit-slider-thumb]:bg-cyan-500
                     [&::-webkit-slider-thumb]:rounded-full"
        />
      </div>

      {/* Divider */}
      <div className="w-px h-6 bg-slate-700" />

      {/* Measure Toggle */}
      <button
        onClick={() => onMeasuringChange?.(!isMeasuring)}
        className={`px-3 py-1.5 rounded text-xs font-medium transition-all flex items-center gap-1.5 ${
          isMeasuring
            ? 'bg-amber-500 text-slate-900'
            : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
        }`}
      >
        <Ruler className="w-3 h-3" />
        {isMeasuring ? 'Measuring' : 'Measure'}
      </button>

      {/* Measurement Display */}
      {currentMeasurement !== null && (
        <span className="text-sm font-mono text-amber-300">
          📏 {currentMeasurement.toFixed(2)} mm
        </span>
      )}
    </div>
  );
}

export default MedicalToolsSidebar;
