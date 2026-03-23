import { Sparkles, Target, Heart, TrendingDown, Move, Ruler, Scissors, Upload, Layers, Scan, FileText } from 'lucide-react';

/**
 * GlassPanel - Reusable glassmorphism panel component
 */
function GlassPanel({ children, className = '' }) {
  return (
    <div
      className={`
        bg-slate-900/90
        backdrop-blur-xl
        rounded-2xl
        border border-slate-700/40
        shadow-2xl shadow-black/30
        ${className}
      `}
    >
      {children}
    </div>
  );
}

/**
 * PACSToolbar - Left sidebar vertical toolbar (PACS workstation style)
 */
export function PACSToolbar({
  onUpload,
  isMeasuring = false,
  onMeasuringChange,
  clippingY = 2,
  onClippingChange,
  showHeatmap = false,
  onHeatmapChange,
  showHeatmapToggle = false,
  onAnalyze,
  analysisStatus = 'idle',
  canAnalyze = false,
  onExportReport,
  canExport = false,
}) {
  const tools = [
    {
      id: 'upload',
      icon: Upload,
      label: 'UPLOAD',
      onClick: onUpload,
      active: false,
      show: true,
    },
    {
      id: 'analyze',
      icon: Scan,
      label: 'ANALYZE',
      onClick: onAnalyze,
      active: analysisStatus === 'success',
      loading: analysisStatus === 'loading',
      show: canAnalyze,
    },
    {
      id: 'slice',
      icon: Scissors,
      label: 'SLICE',
      onClick: () => onClippingChange?.(clippingY >= 2 ? 0 : 2),
      active: clippingY < 2,
      show: true,
    },
    {
      id: 'measure',
      icon: Ruler,
      label: 'MEASURE',
      onClick: () => onMeasuringChange?.(!isMeasuring),
      active: isMeasuring,
      show: true,
    },
    {
      id: 'heatmap',
      icon: Target,
      label: 'HEATMAP',
      onClick: () => onHeatmapChange?.(!showHeatmap),
      active: showHeatmap,
      show: showHeatmapToggle,
    },
    {
      id: 'export',
      icon: FileText,
      label: 'EXPORT',
      onClick: onExportReport,
      active: false,
      show: canExport,
    },
  ];

  return (
    <div className="w-18 h-full bg-slate-900/95 backdrop-blur-xl border-r border-slate-700/30 flex flex-col items-center py-6 gap-2 shadow-xl shadow-black/20">
      {/* Logo */}
      <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center mb-4 shadow-lg shadow-cyan-500/20">
        <Layers className="w-5 h-5 text-white" />
      </div>

      {/* Divider */}
      <div className="w-8 h-px bg-slate-700 mb-3" />

      {/* Tool Buttons */}
      <div className="flex-1 flex flex-col gap-1">
        {tools.filter(t => t.show).map((tool) => (
          <button
            key={tool.id}
            onClick={tool.onClick}
            disabled={tool.loading}
            className={`
              group relative w-12 h-12 rounded-lg flex flex-col items-center justify-center gap-0.5
              transition-all duration-200 
              ${tool.active
                ? 'bg-cyan-500/20 text-cyan-400 ring-1 ring-cyan-500/50'
                : tool.loading
                  ? 'bg-blue-500/20 text-blue-400 cursor-wait'
                  : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/50'
              }
            `}
          >
            {tool.loading ? (
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
            ) : (
              <tool.icon className="w-5 h-5" />
            )}
            <span className="text-[8px] font-medium tracking-wide uppercase">{tool.label}</span>

            {/* Tooltip */}
            <span className="absolute left-full ml-2 px-2 py-1 bg-slate-800 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-50 border border-slate-700">
              {tool.label}
            </span>
          </button>
        ))}
      </div>

      {/* Clipping Slider (vertical) */}
      {clippingY < 2 && (
        <div className="mt-auto pt-3 border-t border-slate-700 w-full flex flex-col items-center gap-2">
          <span className="text-[8px] text-slate-500 uppercase tracking-wide">Clip</span>
          <input
            type="range"
            min="-2"
            max="2"
            step="0.1"
            value={clippingY}
            onChange={(e) => onClippingChange?.(parseFloat(e.target.value))}
            className="w-10 h-1.5 bg-slate-700 rounded-full appearance-none cursor-pointer
                       [&::-webkit-slider-thumb]:appearance-none
                       [&::-webkit-slider-thumb]:w-3
                       [&::-webkit-slider-thumb]:h-3
                       [&::-webkit-slider-thumb]:bg-cyan-400
                       [&::-webkit-slider-thumb]:rounded-full
                       [&::-webkit-slider-thumb]:cursor-pointer
                       [&::-moz-range-thumb]:w-3
                       [&::-moz-range-thumb]:h-3
                       [&::-moz-range-thumb]:bg-cyan-400
                       [&::-moz-range-thumb]:rounded-full
                       [&::-moz-range-thumb]:border-0"
            style={{ writingMode: 'bt-lr', transform: 'rotate(-90deg)', width: '60px' }}
          />
        </div>
      )}

      {/* Status */}
      <div className="mt-auto pt-3">
        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" title="System Ready" />
      </div>
    </div>
  );
}

/**
 * MetricsCard - Top-right floating card for scientific metrics (Medical Imaging Style)
 */
export function MetricsCard({
  riskScore,
  riskLevel,
  healData,
  visible = true
}) {
  if (!visible || (riskScore === null && !healData)) return null;

  const percentage = riskScore !== null ? Math.round(riskScore * 100) : null;

  const getRiskColor = (pct) => {
    if (pct === null) return 'text-slate-400';
    if (pct < 30) return 'text-emerald-400';
    if (pct < 50) return 'text-yellow-400';
    if (pct < 70) return 'text-orange-400';
    return 'text-red-400';
  };

  return (
    <GlassPanel className="p-6 min-w-[240px]">
      {/* Rupture Probability - Large thin typography */}
      {percentage !== null && (
        <div className="mb-5">
          <span className="text-[10px] text-slate-500 uppercase tracking-[0.2em] font-medium block mb-2">
            RUPTURE PROBABILITY
          </span>
          <div className="flex items-baseline gap-1.5">
            <span className={`text-5xl font-extralight tracking-tight tabular-nums ${getRiskColor(percentage)}`}>
              {percentage}
            </span>
            <span className="text-xl font-extralight text-slate-600">%</span>
          </div>
          <span className={`text-[10px] uppercase tracking-[0.15em] font-semibold ${getRiskColor(percentage)} mt-1.5 block`}>
            {riskLevel}
          </span>
        </div>
      )}

      {/* Geometric Metrics - Grid layout for alignment */}
      {healData && (
        <div className="grid grid-cols-1 gap-4 border-t border-slate-700/50 pt-5">
          {healData.max_displacement_mm != null && (
            <div className="grid grid-cols-[1fr_auto] items-baseline gap-2">
              <span className="text-[9px] text-slate-500 uppercase tracking-[0.12em]">
                PEAK DISPLACEMENT
              </span>
              <div className="flex items-baseline gap-1 justify-end">
                <span className="text-lg font-light text-cyan-300 tabular-nums">
                  {healData.max_displacement_mm.toFixed(2)}
                </span>
                <span className="text-[10px] text-slate-500">mm</span>
              </div>
            </div>
          )}
          {healData.mean_displacement_mm != null && (
            <div className="grid grid-cols-[1fr_auto] items-baseline gap-2">
              <span className="text-[9px] text-slate-500 uppercase tracking-[0.12em]">
                MEAN DISPLACEMENT
              </span>
              <div className="flex items-baseline gap-1 justify-end">
                <span className="text-lg font-light text-amber-300 tabular-nums">
                  {healData.mean_displacement_mm.toFixed(2)}
                </span>
                <span className="text-[10px] text-slate-500">mm</span>
              </div>
            </div>
          )}
          {healData.risk_reduction_pct != null && (
            <div className="grid grid-cols-[1fr_auto] items-baseline gap-2">
              <span className="text-[9px] text-slate-500 uppercase tracking-[0.12em]">
                RISK REDUCTION
              </span>
              <div className="flex items-baseline gap-1 justify-end">
                <span className="text-lg font-light text-emerald-300 tabular-nums">
                  {healData.risk_reduction_pct.toFixed(1)}
                </span>
                <span className="text-[10px] text-slate-500">%</span>
              </div>
            </div>
          )}
        </div>
      )}
    </GlassPanel>
  );
}

/**
 * MorphControlBar - Bottom floating island with Pre-Op/Post-Op slider only
 */
export function MorphControlBar({
  morphValue = 0,
  onMorphChange,
  visible = false,
  onHeal,
  healingStatus = 'idle',
  canHeal = false,
}) {
  if (!visible && !canHeal) return null;

  const getRiskState = (t) => {
    if (t < 0.3) return { label: 'ORIGINAL', color: 'text-red-400', bg: 'bg-red-500/20' };
    if (t < 0.7) return { label: 'MORPHING', color: 'text-yellow-400', bg: 'bg-yellow-500/20' };
    return { label: 'HEALED', color: 'text-emerald-400', bg: 'bg-emerald-500/20' };
  };

  const riskState = getRiskState(morphValue);

  return (
    <GlassPanel className="px-8 py-5 flex items-center gap-8">
      {/* Heal Button (if available) */}
      {canHeal && (
        <>
          <button
            onClick={onHeal}
            disabled={healingStatus === 'loading'}
            className={`
              px-5 py-2.5 rounded-lg text-sm font-medium
              flex items-center gap-2 transition-all duration-300
              ${healingStatus === 'loading'
                ? 'bg-emerald-500/50 text-emerald-200 cursor-wait'
                : 'bg-gradient-to-r from-emerald-600 to-cyan-600 text-white hover:from-emerald-500 hover:to-cyan-500 shadow-lg shadow-emerald-500/20'
              }
            `}
          >
            {healingStatus === 'loading' ? (
              <>
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <span>Generating...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                <span>Generate Counterfactual</span>
              </>
            )}
          </button>
          {visible && <div className="w-px h-8 bg-slate-700" />}
        </>
      )}

      {/* Morph Slider */}
      {visible && (
        <div className="flex items-center gap-4">
          <div className="text-right">
            <span className="text-[9px] text-slate-500 uppercase tracking-[0.15em] block">PRE-OP</span>
            <span className="text-xs text-red-400 font-medium">Original</span>
          </div>

          <div className="relative w-48">
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={morphValue}
              onChange={(e) => onMorphChange?.(parseFloat(e.target.value))}
              className="w-full h-2 rounded-full appearance-none cursor-pointer
                         [&::-webkit-slider-thumb]:appearance-none
                         [&::-webkit-slider-thumb]:w-5
                         [&::-webkit-slider-thumb]:h-5
                         [&::-webkit-slider-thumb]:bg-white
                         [&::-webkit-slider-thumb]:rounded-full
                         [&::-webkit-slider-thumb]:shadow-lg
                         [&::-webkit-slider-thumb]:cursor-pointer
                         [&::-webkit-slider-thumb]:border-2
                         [&::-webkit-slider-thumb]:border-slate-300
                         [&::-webkit-slider-thumb]:transition-transform
                         [&::-webkit-slider-thumb]:hover:scale-110
                         [&::-moz-range-thumb]:w-5
                         [&::-moz-range-thumb]:h-5
                         [&::-moz-range-thumb]:bg-white
                         [&::-moz-range-thumb]:rounded-full
                         [&::-moz-range-thumb]:border-0"
              style={{
                background: `linear-gradient(to right, 
                  #ef4444 0%, 
                  #f97316 25%,
                  #eab308 50%, 
                  #84cc16 75%,
                  #22c55e 100%)`
              }}
            />
            {/* Progress indicator */}
            <div className="absolute -bottom-5 left-0 right-0 flex justify-center">
              <span className={`text-[10px] uppercase tracking-wide font-semibold px-2 py-0.5 rounded ${riskState.bg} ${riskState.color}`}>
                {riskState.label}
              </span>
            </div>
          </div>

          <div className="text-left">
            <span className="text-[9px] text-slate-500 uppercase tracking-[0.15em] block">POST-OP</span>
            <span className="text-xs text-emerald-400 font-medium">Healed</span>
          </div>
        </div>
      )}
    </GlassPanel>
  );
}

/**
 * ControlDock - Bottom-center floating dock with main controls (DEPRECATED - use MorphControlBar)
 */
export function ControlDock({
  // Heal button
  onHeal,
  healingStatus = 'idle',
  canHeal = false,

  // Heatmap toggle
  showHeatmap = false,
  onHeatmapChange,
  showHeatmapToggle = false,

  // Morph slider
  morphValue = 0,
  onMorphChange,
  showMorphSlider = false,

  // Measurement
  isMeasuring = false,
  onMeasuringChange,

  // Clipping
  clippingY = 2,
  onClippingChange,
}) {
  // Redirect to new MorphControlBar
  return (
    <MorphControlBar
      morphValue={morphValue}
      onMorphChange={onMorphChange}
      visible={showMorphSlider}
      onHeal={onHeal}
      healingStatus={healingStatus}
      canHeal={canHeal}
    />
  );
}

/**
 * MeasurementIndicator - Floating indicator when measuring
 */
export function MeasurementIndicator({
  isMeasuring,
  pointCount = 0,
  measurement = null
}) {
  if (!isMeasuring && measurement === null) return null;

  return (
    <GlassPanel className="px-4 py-2 flex items-center gap-3">
      <Ruler className="w-4 h-4 text-amber-400" />
      <span className="text-sm text-slate-300 font-light">
        {measurement !== null
          ? <><span className="text-amber-300 font-medium">{measurement.toFixed(2)}</span> mm</>
          : pointCount === 0
            ? 'Click first point'
            : 'Click second point'
        }
      </span>
    </GlassPanel>
  );
}

/**
 * ClippingIndicator - Floating indicator when clipping is active
 */
export function ClippingIndicator({ clippingY }) {
  if (clippingY >= 2) return null;

  const percent = Math.round(((clippingY + 2) / 4) * 100);

  return (
    <GlassPanel className="px-4 py-2 flex items-center gap-2">
      <Scissors className="w-4 h-4 text-cyan-400" />
      <span className="text-sm text-slate-300 font-light">
        Clip <span className="text-cyan-300 font-medium">{percent}%</span>
      </span>
    </GlassPanel>
  );
}

export default {
  GlassPanel,
  PACSToolbar,
  MetricsCard,
  MorphControlBar,
  ControlDock,
  MeasurementIndicator,
  ClippingIndicator
};
