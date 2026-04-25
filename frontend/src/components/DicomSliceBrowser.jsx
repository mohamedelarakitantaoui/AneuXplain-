import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { Layers, Loader2 } from 'lucide-react';

const AXES = [
  { id: 'axial', label: 'Axial' },
  { id: 'sagittal', label: 'Sagittal' },
  { id: 'coronal', label: 'Coronal' },
];

export default function DicomSliceBrowser({ apiUrl, sessionId, axisCounts }) {
  const counts = useMemo(() => ({
    axial: axisCounts?.axial_count ?? 0,
    sagittal: axisCounts?.sagittal_count ?? 0,
    coronal: axisCounts?.coronal_count ?? 0,
  }), [axisCounts]);

  // Per-axis middle-slice defaults. We derive index from state-by-axis so
  // we never need an effect just to reset the slider when the tab changes.
  const [axis, setAxis] = useState('axial');
  const [indexByAxis, setIndexByAxis] = useState({
    axial: Math.floor((counts.axial || 1) / 2),
    sagittal: Math.floor((counts.sagittal || 1) / 2),
    coronal: Math.floor((counts.coronal || 1) / 2),
  });
  const index = indexByAxis[axis] ?? Math.floor((counts[axis] || 1) / 2);
  const [loading, setLoading] = useState(true);

  const max = Math.max(0, (counts[axis] || 1) - 1);
  const url = `${apiUrl}/dicom/slice/${sessionId}/${axis}/${index}`;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className="flex flex-col items-center w-full h-full"
      style={{ color: '#F1F5F9' }}
    >
      {/* Axis tabs */}
      <div
        className="flex"
        style={{
          background: 'rgba(26, 29, 39, 0.9)',
          backdropFilter: 'blur(24px)',
          borderRadius: 24,
          padding: 3,
          border: '1px solid rgba(255,255,255,0.06)',
          marginBottom: 18,
        }}
      >
        {AXES.map((a) => (
          <button
            key={a.id}
            onClick={() => setAxis(a.id)}
            style={{
              padding: '6px 14px',
              borderRadius: 20,
              fontSize: 12,
              fontWeight: 400,
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              background: axis === a.id ? 'rgba(255,255,255,0.1)' : 'transparent',
              color: axis === a.id ? '#F1F5F9' : '#64748B',
              transition: 'all 0.2s ease',
            }}
          >
            <Layers style={{ width: 12, height: 12 }} />
            {a.label}
          </button>
        ))}
      </div>

      {/* Image area */}
      <div
        className="relative flex items-center justify-center"
        style={{
          flex: 1,
          width: '100%',
          minHeight: 0,
          background: '#000',
          borderRadius: 12,
          border: '1px solid rgba(255,255,255,0.06)',
          overflow: 'hidden',
        }}
      >
        {loading && (
          <Loader2
            style={{
              position: 'absolute',
              width: 28,
              height: 28,
              color: '#dc2626',
              animation: 'spin 1s linear infinite',
            }}
          />
        )}
        <img
          key={url}
          src={url}
          alt={`${axis} slice ${index}`}
          onLoadStart={() => setLoading(true)}
          onLoad={() => setLoading(false)}
          onError={() => setLoading(false)}
          style={{
            maxWidth: '100%',
            maxHeight: '100%',
            objectFit: 'contain',
            opacity: loading ? 0 : 1,
            transition: 'opacity 0.15s ease',
          }}
        />
      </div>

      {/* Slider */}
      <div style={{ width: '100%', marginTop: 14, display: 'flex', alignItems: 'center', gap: 12 }}>
        <span style={{ fontSize: 11, color: '#64748B', fontFamily: 'ui-monospace, monospace', minWidth: 52 }}>
          {index + 1} / {max + 1}
        </span>
        <input
          type="range"
          min={0}
          max={max}
          value={index}
          onChange={(e) =>
            setIndexByAxis((prev) => ({ ...prev, [axis]: parseInt(e.target.value, 10) }))
          }
          style={{ flex: 1, accentColor: '#dc2626' }}
        />
      </div>
    </motion.div>
  );
}
