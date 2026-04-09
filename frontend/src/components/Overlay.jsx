import { Ruler, Scissors, Upload, Scan, FileText } from 'lucide-react';

/**
 * PACSToolbar - Left sidebar vertical toolbar (56px, icon-only, clean medical style)
 */
export function PACSToolbar({
  onUpload,
  isMeasuring = false,
  onMeasuringChange,
  clippingY = 2,
  onClippingChange,
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
      label: 'Upload',
      onClick: onUpload,
      active: false,
      show: true,
    },
    {
      id: 'analyze',
      icon: Scan,
      label: 'Analyze',
      onClick: onAnalyze,
      active: analysisStatus === 'success',
      loading: analysisStatus === 'loading',
      show: canAnalyze,
    },
    {
      id: 'slice',
      icon: Scissors,
      label: 'Slice',
      onClick: () => onClippingChange?.(clippingY >= 2 ? 0 : 2),
      active: clippingY < 2,
      show: true,
    },
    {
      id: 'measure',
      icon: Ruler,
      label: 'Measure',
      onClick: () => onMeasuringChange?.(!isMeasuring),
      active: isMeasuring,
      show: true,
    },
    {
      id: 'export',
      icon: FileText,
      label: 'Export',
      onClick: onExportReport,
      active: false,
      show: canExport,
    },
  ];

  return (
    <div
      className="flex flex-col items-center py-5 gap-1 shrink-0"
      style={{
        width: 56,
        height: '100%',
        background: 'rgba(15, 17, 23, 0.85)',
        backdropFilter: 'blur(24px)',
        borderRight: '1px solid rgba(255,255,255,0.06)',
      }}
    >
      {/* Logo mark */}
      <div className="w-8 h-8 rounded-lg flex items-center justify-center mb-4"
        style={{ background: 'linear-gradient(135deg, #4A9EFF, #3B82F6)' }}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2L2 7l10 5 10-5-10-5z" />
          <path d="M2 17l10 5 10-5" />
          <path d="M2 12l10 5 10-5" />
        </svg>
      </div>

      {/* Divider */}
      <div className="w-6 mb-3" style={{ height: 1, background: 'rgba(255,255,255,0.06)' }} />

      {/* Tool Buttons */}
      <div className="flex-1 flex flex-col gap-0.5">
        {tools.filter(t => t.show).map((tool) => (
          <button
            key={tool.id}
            onClick={tool.onClick}
            disabled={tool.loading}
            title={tool.label}
            className="group relative flex items-center justify-center transition-all duration-200"
            style={{
              width: 40,
              height: 40,
              borderRadius: 8,
              marginLeft: 'auto',
              marginRight: 'auto',
              color: tool.active ? '#4A9EFF' : tool.loading ? '#4A9EFF' : '#64748B',
              background: tool.active ? 'rgba(74, 158, 255, 0.08)' : 'transparent',
              borderLeft: tool.active ? '3px solid #4A9EFF' : '3px solid transparent',
              cursor: tool.loading ? 'wait' : 'pointer',
            }}
          >
            {tool.loading ? (
              <svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
            ) : (
              <tool.icon style={{ width: 20, height: 20 }} />
            )}
          </button>
        ))}
      </div>

      {/* Clipping Slider (vertical) */}
      {clippingY < 2 && (
        <div className="mt-auto pt-3 w-full flex flex-col items-center gap-2"
          style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
          <span style={{ fontSize: 9, color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Clip</span>
          <input
            type="range"
            min="-2"
            max="2"
            step="0.1"
            value={clippingY}
            onChange={(e) => onClippingChange?.(parseFloat(e.target.value))}
            className="appearance-none cursor-pointer
                       [&::-webkit-slider-thumb]:appearance-none
                       [&::-webkit-slider-thumb]:w-3
                       [&::-webkit-slider-thumb]:h-3
                       [&::-webkit-slider-thumb]:bg-[#4A9EFF]
                       [&::-webkit-slider-thumb]:rounded-full
                       [&::-webkit-slider-thumb]:cursor-pointer
                       [&::-moz-range-thumb]:w-3
                       [&::-moz-range-thumb]:h-3
                       [&::-moz-range-thumb]:bg-[#4A9EFF]
                       [&::-moz-range-thumb]:rounded-full
                       [&::-moz-range-thumb]:border-0"
            style={{
              writingMode: 'vertical-lr',
              direction: 'rtl',
              width: 60,
              height: 6,
              background: '#242836',
              borderRadius: 3,
              transform: 'rotate(180deg)',
            }}
          />
        </div>
      )}

      {/* Status dot */}
      <div className="mt-auto pt-3">
        <div className="w-1.5 h-1.5 rounded-full" style={{ background: '#34D399' }} title="System Ready" />
      </div>
    </div>
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
    <div
      className="flex items-center gap-2.5 px-3.5 py-2"
      style={{
        background: 'rgba(15, 17, 23, 0.85)',
        backdropFilter: 'blur(24px)',
        borderRadius: 10,
        border: '1px solid rgba(255,255,255,0.06)',
      }}
    >
      <Ruler style={{ width: 14, height: 14, color: '#94A3B8' }} />
      <span style={{ fontSize: 12, color: '#94A3B8', fontWeight: 400 }}>
        {measurement !== null
          ? <><span style={{ color: '#F1F5F9', fontWeight: 400 }}>{measurement.toFixed(2)}</span> mm</>
          : pointCount === 0
            ? 'Click first point'
            : 'Click second point'
        }
      </span>
    </div>
  );
}

/**
 * ClippingIndicator - Floating indicator when clipping is active
 */
export function ClippingIndicator({ clippingY }) {
  if (clippingY >= 2) return null;

  const percent = Math.round(((clippingY + 2) / 4) * 100);

  return (
    <div
      className="flex items-center gap-2 px-3.5 py-2"
      style={{
        background: 'rgba(15, 17, 23, 0.85)',
        backdropFilter: 'blur(24px)',
        borderRadius: 10,
        border: '1px solid rgba(255,255,255,0.06)',
      }}
    >
      <Scissors style={{ width: 14, height: 14, color: '#94A3B8' }} />
      <span style={{ fontSize: 12, color: '#94A3B8', fontWeight: 400 }}>
        Clip <span style={{ color: '#F1F5F9' }}>{percent}%</span>
      </span>
    </div>
  );
}
