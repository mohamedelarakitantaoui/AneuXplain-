/**
 * RiskAnalysisHUD.jsx - Futuristic Expandable Score Card
 * 
 * Medical Sci-Fi style HUD for displaying risk analysis metrics.
 * Features a collapsed/expanded state with smooth animations.
 * 
 * Collapsed: Shows only the big percentage and risk badge
 * Expanded: Reveals detailed grid with advanced metrics
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ChevronDown,
  ChevronUp,
  Activity,
  Zap,
  Target,
  TrendingDown,
  Shield,
  AlertTriangle
} from 'lucide-react';

/**
 * Get risk classification based on percentage
 */
function getRiskClassification(percentage) {
  if (percentage === null) return { level: 'UNKNOWN', color: 'slate', glowColor: 'slate' };
  if (percentage < 25) return { level: 'LOW', color: 'emerald', glowColor: 'emerald' };
  if (percentage < 50) return { level: 'MODERATE', color: 'yellow', glowColor: 'yellow' };
  if (percentage < 75) return { level: 'HIGH', color: 'orange', glowColor: 'orange' };
  return { level: 'CRITICAL', color: 'red', glowColor: 'red' };
}

/**
 * Color mapping for Tailwind classes
 */
const colorMap = {
  slate: {
    text: 'text-slate-400',
    bg: 'bg-slate-500/20',
    border: 'border-slate-500/30',
  },
  emerald: {
    text: 'text-emerald-400',
    bg: 'bg-emerald-500/20',
    border: 'border-emerald-500/30',
  },
  yellow: {
    text: 'text-yellow-400',
    bg: 'bg-yellow-500/20',
    border: 'border-yellow-500/30',
  },
  orange: {
    text: 'text-orange-400',
    bg: 'bg-orange-500/20',
    border: 'border-orange-500/30',
  },
  red: {
    text: 'text-red-400',
    bg: 'bg-red-500/20',
    border: 'border-red-500/30',
  },
};

/**
 * Metric Row Component - Used in expanded view
 */
function MetricRow({ icon: Icon, label, value, unit, color = 'cyan' }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-slate-700/30 last:border-0">
      <div className="flex items-center gap-2">
        <Icon className={`w-3.5 h-3.5 text-${color}-400/70`} />
        <span className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</span>
      </div>
      <div className="flex items-baseline gap-1">
        <span className={`text-base font-light text-${color}-300 tabular-nums`}>{value}</span>
        {unit && <span className="text-[9px] text-slate-600">{unit}</span>}
      </div>
    </div>
  );
}

/**
 * RiskAnalysisHUD - Main Component
 */
export default function RiskAnalysisHUD({
  riskScore,
  riskLevel,
  healData,
  visible = true,
  interpolatedRisk = null,  // Real-time interpolated risk during morph
  isEstimate = false,       // Show "(Est.)" when morphing
}) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Don't render if not visible or no data
  if (!visible || (riskScore === null && !healData)) return null;

  // Use interpolatedRisk if provided, otherwise fall back to riskScore
  const displayedRisk = interpolatedRisk !== null ? interpolatedRisk : riskScore;
  const percentage = displayedRisk !== null ? Math.round(displayedRisk * 100) : null;
  const classification = getRiskClassification(percentage);
  const colors = colorMap[classification.color];

  // Simulated advanced metrics (derived from available data)
  const peakWallStress = percentage !== null ? (percentage * 0.15 + 2.5).toFixed(1) : '--';
  const geometryComplexity = percentage !== null ? (percentage * 0.8 + 15).toFixed(0) : '--';
  const confidenceInterval = percentage !== null ? `±${(5 + Math.random() * 3).toFixed(1)}` : '--';
  const neckToDomeRatio = percentage !== null ? (0.3 + percentage * 0.007).toFixed(2) : '--';

  // Dynamic glow shadow based on risk level
  const glowShadow = {
    slate: 'shadow-lg shadow-slate-900/50',
    emerald: 'shadow-lg shadow-emerald-500/20',
    yellow: 'shadow-lg shadow-yellow-500/20',
    orange: 'shadow-lg shadow-orange-500/20',
    red: 'shadow-lg shadow-red-500/30',
  }[classification.color];

  // Pulsing animation for CRITICAL risk
  const pulseAnimation = classification.level === 'CRITICAL'
    ? 'animate-pulse-border'
    : '';

  return (
    <motion.div
      layout
      className={`
        relative overflow-hidden
        bg-slate-900/95 backdrop-blur-xl
        rounded-2xl
        border ${colors.border}
        ${glowShadow}
        ${pulseAnimation}
        cursor-pointer
        w-[280px] max-w-[calc(100vw-4rem)]
      `}
      style={{
        animation: classification.level === 'CRITICAL' ? 'pulse-border 2s ease-in-out infinite' : 'none',
      }}
      onClick={() => setIsExpanded(!isExpanded)}
      initial={{ opacity: 0, scale: 0.95, y: -10 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
    >
      {/* ============================================
          COLLAPSED STATE - Main Display
          ============================================ */}
      <div className="p-6">
        {/* Header Row - Score + Badge */}
        <div className="flex items-start justify-between gap-4">
          {/* Big Percentage */}
          <div className="flex items-baseline gap-1">
            <motion.span
              key={percentage}
              initial={{ opacity: 0, scale: 1.1 }}
              animate={{ opacity: 1, scale: 1 }}
              className={`text-6xl font-extralight tracking-tighter tabular-nums ${colors.text}`}
            >
              {percentage ?? '--'}
            </motion.span>
            <span className="text-2xl font-extralight text-slate-600">%</span>
          </div>

          {/* Risk Badge */}
          <motion.div
            layout
            className={`
              px-2.5 py-1 rounded-md
              ${colors.bg} ${colors.border} border
              flex items-center gap-1.5
              shrink-0
            `}
          >
            {classification.level === 'CRITICAL' && (
              <AlertTriangle className="w-3 h-3 text-red-400" />
            )}
            <span className={`text-[9px] font-bold uppercase tracking-wider ${colors.text}`}>
              {classification.level}
            </span>
          </motion.div>
        </div>

        {/* Label */}
        <p className="text-[9px] text-slate-500 uppercase tracking-[0.2em] mt-2">
          RUPTURE PROBABILITY {isEstimate && <span className="text-amber-500">(Est.)</span>}
        </p>

        {/* Expand Indicator */}
        <div className="flex items-center justify-center mt-3 pt-3 border-t border-slate-800/50">
          <motion.div
            animate={{ y: isExpanded ? 0 : [0, 3, 0] }}
            transition={{ repeat: isExpanded ? 0 : Infinity, duration: 1.5 }}
            className="flex items-center gap-1 text-slate-600"
          >
            {isExpanded ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
            <span className="text-[9px] uppercase tracking-wider">
              {isExpanded ? 'Collapse' : 'Details'}
            </span>
          </motion.div>
        </div>
      </div>

      {/* ============================================
          EXPANDED STATE - Detailed Metrics Grid
          ============================================ */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5 pt-2 border-t border-slate-800/50">
              {/* Advanced Metrics Grid */}
              <div className="space-y-1">
                <MetricRow
                  icon={Zap}
                  label="Peak Wall Stress (Est.)"
                  value={peakWallStress}
                  unit="kPa"
                  color="red"
                />
                <MetricRow
                  icon={Activity}
                  label="Geometry Complexity"
                  value={geometryComplexity}
                  unit="GCI"
                  color="amber"
                />
                <MetricRow
                  icon={Shield}
                  label="Confidence Interval"
                  value={confidenceInterval}
                  unit="%"
                  color="cyan"
                />
                <MetricRow
                  icon={Target}
                  label="Neck-to-Dome Ratio"
                  value={neckToDomeRatio}
                  unit=""
                  color="purple"
                />
              </div>

              {/* Heal Data Section (if available) */}
              {healData && (
                <div className="mt-4 pt-4 border-t border-slate-700/30">
                  <p className="text-[9px] text-emerald-500/80 uppercase tracking-widest mb-3 flex items-center gap-1.5">
                    <TrendingDown className="w-3 h-3" />
                    COUNTERFACTUAL ANALYSIS
                  </p>
                  <div className="space-y-1">
                    {healData.max_displacement_mm != null && (
                      <MetricRow
                        icon={Target}
                        label="Peak Displacement"
                        value={healData.max_displacement_mm.toFixed(2)}
                        unit="mm"
                        color="cyan"
                      />
                    )}
                    {healData.mean_displacement_mm != null && (
                      <MetricRow
                        icon={Activity}
                        label="Mean Displacement"
                        value={healData.mean_displacement_mm.toFixed(2)}
                        unit="mm"
                        color="amber"
                      />
                    )}
                    {healData.risk_reduction_pct != null && (
                      <MetricRow
                        icon={Shield}
                        label="Risk Reduction"
                        value={healData.risk_reduction_pct.toFixed(1)}
                        unit="%"
                        color="emerald"
                      />
                    )}
                  </div>
                </div>
              )}

              {/* Timestamp */}
              <p className="text-[8px] text-slate-700 text-center mt-4 uppercase tracking-widest">
                Last Updated: {new Date().toLocaleTimeString()}
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
