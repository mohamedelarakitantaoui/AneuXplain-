/**
 * RiskScoreCard.jsx - Clean floating risk score display
 * Minimal vital-sign-monitor style card for the 3D viewport
 */

import { motion } from 'framer-motion';

const RISK_COLORS = {
  LOW:      '#34D399',
  MODERATE: '#FBBF24',
  HIGH:     '#F87171',
  CRITICAL: '#EF4444',
};

function getRiskLevel(pct) {
  if (pct < 25) return 'LOW';
  if (pct < 50) return 'MODERATE';
  if (pct < 75) return 'HIGH';
  return 'CRITICAL';
}

export default function RiskScoreCard({ riskScore, riskLevel }) {
  if (riskScore === null && riskScore === undefined) return null;

  const percentage = Math.round((riskScore ?? 0) * 100);
  const level = riskLevel || getRiskLevel(percentage);
  const color = RISK_COLORS[level] || RISK_COLORS.LOW;

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      style={{
        width: 240,
        background: 'rgba(255,255,255,0.88)',
        backdropFilter: 'blur(24px)',
        WebkitBackdropFilter: 'blur(24px)',
        borderRadius: 20,
        border: '1px solid rgba(0,0,0,0.08)',
        padding: '20px 24px',
        boxShadow: '0 8px 40px rgba(0,0,0,0.13), 0 2px 8px rgba(0,0,0,0.06)',
      }}
    >
      {/* Risk percentage */}
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 3 }}>
        <span
          style={{
            fontSize: 48,
            fontWeight: 300,
            lineHeight: 1,
            color: color,
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          {percentage}
        </span>
        <span style={{ fontSize: 20, fontWeight: 300, color: '#64748B' }}>%</span>
      </div>

      {/* Label */}
      <p style={{
        fontSize: 10,
        color: '#64748B',
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        marginTop: 4,
      }}>
        Rupture Probability
      </p>

      {/* Risk level badge */}
      <div
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
          marginTop: 10,
          padding: '3px 10px',
          borderRadius: 20,
          background: `${color}18`,
          fontSize: 10,
          fontWeight: 500,
          color: color,
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
        }}
      >
        {level === 'CRITICAL' && (
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: color, boxShadow: `0 0 8px ${color}` }} />
        )}
        {level}
      </div>

      {/* Divider + timestamp */}
      <div style={{ height: 1, background: 'rgba(0,0,0,0.07)', margin: '14px 0 10px' }} />
      <p style={{ fontSize: 10, color: '#94A3B8', letterSpacing: '0.02em' }}>
        Analyzed at {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
      </p>
    </motion.div>
  );
}
