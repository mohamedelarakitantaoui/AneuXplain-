import { Ruler, Scissors, Upload, Scan, FileText } from 'lucide-react';

/**
 * PACSToolbar - Floating circles on the left edge of the viewport
 */
export function PACSToolbar({
  onUpload,
  onLogoClick,
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
  const sliceActive = clippingY < 2;

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
      onClick: () => onClippingChange?.(sliceActive ? 2 : 0),
      active: sliceActive,
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

  const circleBase = {
    width: 46,
    height: 46,
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backdropFilter: 'blur(20px)',
    WebkitBackdropFilter: 'blur(20px)',
    transition: 'all 0.2s ease',
    cursor: 'pointer',
    border: 'none',
    outline: 'none',
  };

  return (
    <>
      {/* Floating circles column */}
      <div
        style={{
          position: 'fixed',
          left: 16,
          top: '50%',
          transform: 'translateY(-50%)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 10,
          zIndex: 100,
        }}
      >
        {/* Logo circle — click opens landing page */}
        <button
          onClick={onLogoClick}
          title="About AneuXplain"
          style={{
            ...circleBase,
            background: '#ffffff',
            boxShadow: '0 2px 8px rgba(0,0,0,0.10), 0 0 0 1px rgba(0,0,0,0.06)',
            cursor: 'pointer',
            marginBottom: 6,
            transition: 'box-shadow 0.2s ease, transform 0.15s ease',
            overflow: 'hidden',
          }}
          onMouseEnter={e => { e.currentTarget.style.boxShadow = '0 4px 16px rgba(0,0,0,0.16)'; e.currentTarget.style.transform = 'scale(1.07)'; }}
          onMouseLeave={e => { e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.10), 0 0 0 1px rgba(0,0,0,0.06)'; e.currentTarget.style.transform = 'scale(1)'; }}
        >
          <img src="/logoAneuX.png" alt="AneuXplain" style={{ width: 32, height: 32, objectFit: 'contain' }} />
        </button>

        {/* Thin separator */}
        <div style={{ width: 1, height: 16, background: 'rgba(0,0,0,0.12)', marginBottom: 2 }} />

        {/* Tool buttons */}
        {tools.filter(t => t.show).map((tool) => (
          <button
            key={tool.id}
            onClick={tool.onClick}
            disabled={tool.loading}
            title={tool.label}
            style={{
              ...circleBase,
              background: tool.active
                ? 'rgba(220, 38, 38, 0.08)'
                : '#ffffff',
              boxShadow: tool.active
                ? '0 0 0 1.5px rgba(220, 38, 38, 0.45), 0 4px 14px rgba(220, 38, 38, 0.1)'
                : '0 2px 8px rgba(0,0,0,0.10), 0 0 0 1px rgba(0,0,0,0.06)',
              color: tool.active ? '#dc2626' : tool.loading ? '#dc2626' : '#94A3B8',
              cursor: tool.loading ? 'wait' : 'pointer',
            }}
          >
            {tool.loading ? (
              <svg className="animate-spin" width="18" height="18" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
            ) : (
              <tool.icon style={{ width: 18, height: 18 }} />
            )}
          </button>
        ))}

        {/* Bottom separator + status dot */}
        <div style={{ width: 1, height: 16, background: 'rgba(0,0,0,0.12)', marginTop: 2 }} />
        <div
          style={{ width: 7, height: 7, borderRadius: '50%', background: '#34D399', boxShadow: '0 0 8px rgba(52, 211, 153, 0.5)' }}
          title="System Ready"
        />
      </div>

      {/* Clipping slider — floats to the right of the circles when slice is active */}
      {sliceActive && (
        <div
          style={{
            position: 'fixed',
            left: 74,
            top: '50%',
            transform: 'translateY(-50%)',
            background: '#ffffff',
            backdropFilter: 'blur(20px)',
            WebkitBackdropFilter: 'blur(20px)',
            borderRadius: 14,
            padding: '14px 10px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.10), 0 0 0 1px rgba(0,0,0,0.06)',
            zIndex: 100,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 8,
          }}
        >
          <span style={{ fontSize: 9, color: '#dc2626', textTransform: 'uppercase', letterSpacing: '0.12em', fontWeight: 500 }}>Clip</span>
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
                       [&::-webkit-slider-thumb]:bg-[#dc2626]
                       [&::-webkit-slider-thumb]:rounded-full
                       [&::-webkit-slider-thumb]:cursor-pointer
                       [&::-moz-range-thumb]:w-3
                       [&::-moz-range-thumb]:h-3
                       [&::-moz-range-thumb]:bg-[#dc2626]
                       [&::-moz-range-thumb]:rounded-full
                       [&::-moz-range-thumb]:border-0"
            style={{
              writingMode: 'vertical-lr',
              direction: 'rtl',
              width: 80,
              height: 6,
              background: 'rgba(220, 38, 38, 0.12)',
              borderRadius: 3,
              transform: 'rotate(180deg)',
            }}
          />
          <span style={{ fontSize: 9, color: '#64748B', fontFamily: 'monospace' }}>
            {Math.round(((clippingY + 2) / 4) * 100)}%
          </span>
        </div>
      )}
    </>
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
        background: 'rgba(255,255,255,0.92)',
        backdropFilter: 'blur(24px)',
        borderRadius: 10,
        border: '1px solid rgba(0,0,0,0.08)',
        boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
      }}
    >
      <Ruler style={{ width: 14, height: 14, color: '#94A3B8' }} />
      <span style={{ fontSize: 12, color: '#94A3B8', fontWeight: 400 }}>
        {measurement !== null
          ? <><span style={{ color: '#0F1117', fontWeight: 500 }}>{measurement.toFixed(2)}</span> mm</>
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
        background: 'rgba(255,255,255,0.92)',
        backdropFilter: 'blur(24px)',
        borderRadius: 10,
        border: '1px solid rgba(0,0,0,0.08)',
        boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
      }}
    >
      <Scissors style={{ width: 14, height: 14, color: '#94A3B8' }} />
      <span style={{ fontSize: 12, color: '#94A3B8', fontWeight: 400 }}>
        Clip <span style={{ color: '#0F1117', fontWeight: 500 }}>{percent}%</span>
      </span>
    </div>
  );
}
