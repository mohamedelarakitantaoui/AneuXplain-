/**
 * MorphologyReport.jsx - Clinical Morphological Analysis Panel
 * Clean medical design — lab-report style parameter cards with
 * reference range bars, risk badges, and expandable explanations.
 */

import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// ============================================
// Design tokens
// ============================================
const COLORS = {
  bg:        'rgba(255,255,255,0.88)',
  surface:   'rgba(255,255,255,0.72)',
  border:    'rgba(0,0,0,0.07)',
  textPri:   '#0F1117',
  textSec:   '#475569',
  textTer:   '#94A3B8',
  accent:    '#dc2626',
};

const GLASS = {
  background:              COLORS.bg,
  backdropFilter:          'blur(24px)',
  WebkitBackdropFilter:    'blur(24px)',
  borderRadius:            20,
  border:                  '1px solid rgba(0,0,0,0.08)',
  boxShadow:               '0 8px 40px rgba(0,0,0,0.13), 0 2px 8px rgba(0,0,0,0.06)',
  width:                   320,
  maxHeight:               'calc(100vh - 88px)',
  display:                 'flex',
  flexDirection:           'column',
  overflow:                'hidden',
};

const RISK_COLORS = {
  LOW:      '#34D399',
  MODERATE: '#FBBF24',
  HIGH:     '#F87171',
  'N/A':    '#64748B',
};

// ============================================
// Threshold lookup for the range bar
// ============================================
const THRESHOLDS = {
  'Aspect Ratio':       { lowUpper: 1.0,   highLower: 1.6,   invert: false, unit: '' },
  'Dome-to-Neck Ratio': { lowUpper: 1.0,   highLower: 2.0,   invert: false, unit: '' },
  'Irregularity Index':  { lowUpper: 0.1,   highLower: 0.25,  invert: false, unit: '' },
  'Neck Width':          { lowUpper: 4.0,   highLower: 2.0,   invert: true,  unit: 'mm' },
  'Dome Height':         { lowUpper: 5.0,   highLower: 10.0,  invert: false, unit: 'mm' },
  'Volume':              { lowUpper: 100.0, highLower: 500.0, invert: false, unit: 'mm³' },
  'Size Ratio':          { lowUpper: 2.0,   highLower: 4.0,   invert: false, unit: '' },
};


// ============================================
// Skeleton loading
// ============================================
function SkeletonCard() {
  return (
    <div style={{
      background: COLORS.surface,
      borderRadius: 12,
      padding: 16,
      marginBottom: 12,
      border: '1px solid rgba(0,0,0,0.06)',
    }}>
      <div style={{ height: 10, width: 100, background: '#E2E8F0', borderRadius: 4, marginBottom: 12 }} className="animate-pulse" />
      <div style={{ height: 28, width: 70, background: '#E2E8F0', borderRadius: 4, marginBottom: 8 }} className="animate-pulse" />
      <div style={{ height: 4, width: '100%', background: '#E2E8F0', borderRadius: 2 }} className="animate-pulse" />
    </div>
  );
}

function SkeletonSummary() {
  return (
    <div style={{
      background: COLORS.surface,
      borderRadius: 12,
      padding: 16,
      marginBottom: 12,
      border: '1px solid rgba(0,0,0,0.06)',
    }}>
      <div className="flex gap-4 mb-3">
        <div style={{ height: 36, width: 50, background: '#E2E8F0', borderRadius: 4 }} className="animate-pulse" />
        <div className="flex-1 space-y-2">
          <div style={{ height: 14, width: 80, background: '#E2E8F0', borderRadius: 4 }} className="animate-pulse" />
          <div style={{ height: 10, width: 120, background: '#E2E8F0', borderRadius: 4 }} className="animate-pulse" />
        </div>
      </div>
      <div className="space-y-2">
        <div style={{ height: 8, width: '100%', background: '#E2E8F0', borderRadius: 3 }} className="animate-pulse" />
        <div style={{ height: 8, width: '85%', background: '#E2E8F0', borderRadius: 3 }} className="animate-pulse" />
      </div>
    </div>
  );
}


// ============================================
// Reference range bar (lab-report style)
// ============================================
function RangeBar({ value, lowUpper, highLower, invert }) {
  if (value == null) return null;

  const rangeMax = highLower * 2.5 || 10;
  const clampedValue = Math.max(0, Math.min(value, rangeMax));
  const markerPct = (clampedValue / rangeMax) * 100;
  const lowPct = (lowUpper / rangeMax) * 100;
  const highPct = (highLower / rangeMax) * 100;

  const greenZone = invert
    ? { left: `${lowPct}%`, right: 0 }
    : { left: 0, width: `${lowPct}%` };
  const amberZone = invert
    ? { left: `${highPct}%`, width: `${lowPct - highPct}%` }
    : { left: `${lowPct}%`, width: `${highPct - lowPct}%` };
  const redZone = invert
    ? { left: 0, width: `${highPct}%` }
    : { left: `${highPct}%`, right: 0 };

  return (
    <div style={{ marginTop: 8, marginBottom: 4 }}>
      <div style={{
        position: 'relative',
        height: 4,
        borderRadius: 2,
        background: '#E2E8F0',
        overflow: 'visible',
      }}>
        {/* Green zone */}
        <div style={{
          position: 'absolute', top: 0, bottom: 0,
          ...greenZone,
          background: 'rgba(52, 211, 153, 0.3)',
          borderRadius: '2px 0 0 2px',
        }} />
        {/* Amber zone */}
        <div style={{
          position: 'absolute', top: 0, bottom: 0,
          ...amberZone,
          background: 'rgba(251, 191, 36, 0.3)',
        }} />
        {/* Red zone */}
        <div style={{
          position: 'absolute', top: 0, bottom: 0,
          ...redZone,
          background: 'rgba(248, 113, 113, 0.3)',
          borderRadius: '0 2px 2px 0',
        }} />
        {/* Marker triangle */}
        <div style={{
          position: 'absolute',
          top: -3,
          left: `${markerPct}%`,
          transform: 'translateX(-50%)',
          width: 0,
          height: 0,
          borderLeft: '4px solid transparent',
          borderRight: '4px solid transparent',
          borderTop: '5px solid #334155',
        }} />
      </div>
    </div>
  );
}


// ============================================
// Parameter card
// ============================================
function ParameterCard({ param, isExpanded, isSelected, onToggle }) {
  const isNA = param.risk_level === 'N/A' || param.value == null;
  const riskColor = isNA ? '#64748B' : (RISK_COLORS[param.risk_level] || RISK_COLORS['N/A']);
  const thresh = THRESHOLDS[param.parameter];

  const formatValue = (val) => {
    if (val == null) return 'N/A';
    if (typeof val === 'number') {
      return val < 1 ? val.toFixed(3) : val < 100 ? val.toFixed(2) : val.toFixed(1);
    }
    return val;
  };

  return (
    <motion.div
      layout
      onClick={isNA ? undefined : onToggle}
      style={{
        background: COLORS.surface,
        borderRadius: 12,
        padding: 16,
        marginBottom: 12,
        cursor: isNA ? 'default' : 'pointer',
        borderLeft: isExpanded || isSelected ? `3px solid ${riskColor}` : '3px solid transparent',
        border: `1px solid rgba(0,0,0,0.06)`,
        borderLeftWidth: isExpanded || isSelected ? 3 : 1,
        borderLeftColor: isExpanded || isSelected ? riskColor : 'rgba(0,0,0,0.06)',
        transition: 'border-color 0.2s, transform 0.2s, opacity 0.2s',
        transform: 'translateY(0)',
        opacity: isNA ? 0.6 : 1,
      }}
      whileHover={isNA ? {} : { y: -1 }}
    >
      {/* Top row: parameter name + risk dot */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <span style={{
          fontSize: 11,
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
          color: COLORS.textSec,
          fontWeight: 400,
        }}>
          {param.parameter}
        </span>
        <div style={{
          width: 8,
          height: 8,
          borderRadius: '50%',
          background: riskColor,
          flexShrink: 0,
        }} />
      </div>

      {/* Value row */}
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
        <span style={{
          fontSize: 28,
          fontWeight: 300,
          fontVariantNumeric: 'tabular-nums',
          color: isNA ? COLORS.textTer : COLORS.textPri,
          lineHeight: 1.1,
        }}>
          {formatValue(param.value)}
        </span>
        {param.unit && !isNA && (
          <span style={{ fontSize: 12, color: COLORS.textTer }}>{param.unit}</span>
        )}
      </div>

      {/* Range bar */}
      {thresh && !isNA && (
        <RangeBar
          value={param.value}
          lowUpper={thresh.lowUpper}
          highLower={thresh.highLower}
          invert={thresh.invert}
        />
      )}

      {/* Normal range text or N/A info */}
      {isNA ? (
        <p style={{ fontSize: 10, color: COLORS.textTer, marginTop: 6, fontStyle: 'italic' }}>
          Requires parent vessel reference
        </p>
      ) : (
        <p style={{ fontSize: 10, color: COLORS.textTer, marginTop: 4 }}>
          Normal: {param.normal_range}
        </p>
      )}

      {/* Expandable explanation */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            style={{ overflow: 'hidden' }}
          >
            <div style={{ marginTop: 12, paddingTop: 12, borderTop: `1px solid ${COLORS.border}` }}>
              {/* Clinical Finding */}
              <div style={{ marginBottom: 10 }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  marginBottom: 4,
                }}>
                  <div style={{ width: 3, height: 12, borderRadius: 2, background: riskColor }} />
                  <span style={{
                    fontSize: 10,
                    textTransform: 'uppercase',
                    letterSpacing: '0.1em',
                    color: COLORS.textTer,
                  }}>
                    Clinical Finding
                  </span>
                </div>
                <p style={{ fontSize: 12, color: COLORS.textSec, lineHeight: 1.6, marginLeft: 9 }}>
                  {param.explanation}
                </p>
              </div>

              {/* Significance */}
              {param.clinical_significance && (
                <div>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    marginBottom: 4,
                  }}>
                    <div style={{ width: 3, height: 12, borderRadius: 2, background: COLORS.accent }} />
                    <span style={{
                      fontSize: 10,
                      textTransform: 'uppercase',
                      letterSpacing: '0.1em',
                      color: COLORS.textTer,
                    }}>
                      Significance
                    </span>
                  </div>
                  <p style={{ fontSize: 12, color: '#64748B', lineHeight: 1.6, marginLeft: 9 }}>
                    {param.clinical_significance}
                  </p>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}


// ============================================
// Summary card
// ============================================
function SummaryCard({ clinicalReport }) {
  const [showFull, setShowFull] = useState(false);
  const summary = clinicalReport.summary || '';
  const isLong = summary.length > 180;

  return (
    <div style={{
      background: COLORS.surface,
      borderRadius: 12,
      padding: 16,
      marginBottom: 12,
      border: '1px solid rgba(0,0,0,0.06)',
    }}>
      {/* Mini stat row */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 12 }}>
        {clinicalReport.high_risk_count > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{ width: 6, height: 6, borderRadius: '50%', background: RISK_COLORS.HIGH }} />
            <span style={{ fontSize: 12, color: COLORS.textSec, fontWeight: 400 }}>
              {clinicalReport.high_risk_count} High
            </span>
          </div>
        )}
        {clinicalReport.moderate_risk_count > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{ width: 6, height: 6, borderRadius: '50%', background: RISK_COLORS.MODERATE }} />
            <span style={{ fontSize: 12, color: COLORS.textSec, fontWeight: 400 }}>
              {clinicalReport.moderate_risk_count} Mod
            </span>
          </div>
        )}
        {clinicalReport.low_risk_count > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{ width: 6, height: 6, borderRadius: '50%', background: RISK_COLORS.LOW }} />
            <span style={{ fontSize: 12, color: COLORS.textSec, fontWeight: 400 }}>
              {clinicalReport.low_risk_count} Low
            </span>
          </div>
        )}
      </div>

      {/* Summary text */}
      <p style={{
        fontSize: 12,
        color: COLORS.textSec,
        lineHeight: 1.65,
        display: '-webkit-box',
        WebkitLineClamp: showFull ? 'unset' : 4,
        WebkitBoxOrient: 'vertical',
        overflow: showFull ? 'visible' : 'hidden',
      }}>
        {summary}
      </p>
      {isLong && (
        <button
          onClick={(e) => { e.stopPropagation(); setShowFull(!showFull); }}
          style={{
            fontSize: 11,
            color: COLORS.accent,
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            marginTop: 4,
            padding: 0,
          }}
        >
          {showFull ? 'Show less' : 'Read more'}
        </button>
      )}
    </div>
  );
}


// ============================================
// Main MorphologyReport component
// ============================================
export default function MorphologyReport({
  clinicalReport,
  onParameterSelect,
  isLoading = false,
}) {
  const [expandedParam, setExpandedParam] = useState(null);

  const handleParamToggle = useCallback((paramName, spatialData) => {
    console.log('[MorphologyReport] Parameter selected:', paramName, spatialData);
    setExpandedParam(prev => {
      const next = prev === paramName ? null : paramName;
      if (onParameterSelect) {
        onParameterSelect(next ? paramName : null, next ? spatialData : null);
      }
      return next;
    });
  }, [onParameterSelect]);

  // Loading state
  if (isLoading) {
    return (
      <motion.div
        initial={{ opacity: 0, x: 16 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
        style={GLASS}
      >
        <div style={{ padding: '16px 20px', borderBottom: `1px solid ${COLORS.border}` }}>
          <div style={{ height: 14, width: 140, background: '#E2E8F0', borderRadius: 4 }} className="animate-pulse" />
          <div style={{ height: 10, width: 100, background: '#E2E8F0', borderRadius: 4, marginTop: 6 }} className="animate-pulse" />
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: 16 }}>
          <SkeletonSummary />
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </div>
      </motion.div>
    );
  }

  if (!clinicalReport || !clinicalReport.parameters) return null;

  const paramCount = clinicalReport.parameters.filter(p => p.risk_level !== 'N/A' && p.value != null).length;

  return (
    <motion.div
      initial={{ opacity: 0, x: 16 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      style={GLASS}
    >
      {/* Header */}
      <div style={{ padding: '16px 20px', borderBottom: `1px solid ${COLORS.border}`, flexShrink: 0 }}>
        <h2 style={{
          fontSize: 13,
          fontWeight: 500,
          color: COLORS.textPri,
          letterSpacing: '-0.01em',
        }}>
          Morphological Analysis
        </h2>
        <p style={{
          fontSize: 11,
          color: COLORS.textTer,
          marginTop: 2,
        }}>
          {paramCount} parameters analyzed
        </p>
      </div>

      {/* Scrollable body */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: 16,
      }}>
        <SummaryCard clinicalReport={clinicalReport} />

        {clinicalReport.parameters.map((param) => (
          <ParameterCard
            key={param.parameter}
            param={param}
            isExpanded={expandedParam === param.parameter}
            isSelected={expandedParam === param.parameter}
            onToggle={() => handleParamToggle(param.parameter, param.spatial)}
          />
        ))}

        <div style={{ height: 8 }} />
      </div>
    </motion.div>
  );
}
